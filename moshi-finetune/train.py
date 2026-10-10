import dataclasses
import logging
import os
import sys
import shutil
import contextlib
from contextlib import ExitStack

# Ensure local moshi source takes precedence over installed site-packages
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import pprint
from pathlib import Path

# Personaplex rope.py의 apply_rope가 @torch.compile로 감싸져 있는데,
# gradient checkpointing + FSDP와 함께 쓰면 inductor가 FakeTensor 충돌을 일으킴.
# 학습에서는 torch.compile을 비활성화하고 eager 모드로 실행합니다.
import torch._dynamo
torch._dynamo.config.suppress_errors = True  # compile 실패 시 eager로 fallback
os.environ["TORCHDYNAMO_DISABLE"] = "1"      # torch.compile 전역 비활성화

import fire
import torch
import torch.cuda
import torch.distributed as dist
import safetensors.torch
from torch.optim import AdamW, lr_scheduler

from torch.nn import functional as F

from finetune.args import TrainArgs
from finetune.checkpointing import Checkpointer
from finetune.fusion_dynamics import FusionDynamicsMonitor
from finetune.data.data_loader import build_data_loader
from finetune.data.interleaver import InterleavedTokenizer, Interleaver
from finetune.distributed import (
    BACKEND,
    avg_aggregate,
    get_rank,
    get_world_size,
    is_torchrun,
    set_device,
)
from finetune.eval import evaluate
from finetune.loss import (
    bc_head_confusion_counts,
    build_bc_targets,
    compute_loss_with_mask,
    compute_face_loss,
    epad_confusion_counts,
    epad_metrics_from_counts,
    turn_end_mask,
)
from finetune.monitoring.metrics_logger import (
    MetricsLogger,
    eval_log_msg,
    get_eval_logs,
    get_train_logs,
    train_log_msg,
)
from finetune.monitoring.utils import set_logger
from finetune.utils import TrainState, logged_closing, set_random_seed
from finetune.wrapped_model import get_fsdp_model

# personaplex_MBG/moshi 패키지 (Python 모듈명: moshi)
from moshi.models.loaders import (
    get_mimi,
    get_moshi_lm,
    _lm_kwargs,
    MIMI_NAME,
    MOSHI_NAME,
    TEXT_TOKENIZER_NAME,
    DEFAULT_REPO,
)

logger = logging.getLogger("train")


def main_logger_info(message: str) -> None:
    if get_rank() == 0:
        logger.info(message)


def _looks_like_hf_repo_id(value: str) -> bool:
    """'org/name' 형태의 HuggingFace repo id 인지 판별."""
    return (
        not os.path.isabs(value)
        and not value.startswith((".", "~"))
        and value.count("/") == 1
        and not value.endswith((".safetensors", ".pt", ".ckpt", ".bin"))
    )


def _download_moshi_weights(repo_id: str) -> str:
    """HF 레포에서 model.safetensors 다운로드 후 로컬 캐시 경로 반환."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "HF repo id를 사용하려면 huggingface_hub가 필요합니다 "
            "(pip install huggingface_hub)."
        ) from e
    try:
        return hf_hub_download(repo_id, MOSHI_NAME)
    except Exception as e:
        raise RuntimeError(
            f"{repo_id} 에서 {MOSHI_NAME} 다운로드 실패: {e}"
        ) from e


def _resolve_teacher_checkpoint(cl_args, moshi_path: str | None) -> str:
    """continual_learning teacher 가중치 경로를 해석한다.

    우선순위: continual_learning.hf_repo_id > continual_learning.teacher_checkpoint
    > moshi_paths 로 이미 해석된 moshi_path.
    teacher_checkpoint 에는 로컬 파일, 체크포인트 디렉터리, 또는
    "nvidia/personaplex-7b-v1" 같은 HF repo id 를 모두 넣을 수 있다.
    """
    if cl_args.hf_repo_id:
        main_logger_info(
            f"continual_learning teacher를 HF 레포에서 받는 중: {cl_args.hf_repo_id}"
        )
        return _download_moshi_weights(cl_args.hf_repo_id)

    candidate = cl_args.teacher_checkpoint
    if not candidate:
        # 기본값: 학생 모델과 동일한 원본 PersonaPlex 가중치.
        if not moshi_path or not os.path.isfile(moshi_path):
            raise FileNotFoundError(
                "continual_learning teacher checkpoint을 찾을 수 없습니다 "
                f"(moshi_paths.moshi_path={moshi_path!r}). "
                "config의 continual_learning에 hf_repo_id 또는 "
                "teacher_checkpoint를 설정하세요."
            )
        return moshi_path

    if os.path.isfile(candidate):
        return candidate
    if os.path.isdir(candidate):
        local_weights = os.path.join(candidate, MOSHI_NAME)
        if os.path.isfile(local_weights):
            return local_weights
        raise FileNotFoundError(
            f"continual_learning teacher 디렉터리에 {MOSHI_NAME}가 없습니다: "
            f"{candidate!r}"
        )
    if _looks_like_hf_repo_id(candidate):
        main_logger_info(
            f"continual_learning teacher를 HF 레포에서 받는 중: {candidate}"
        )
        return _download_moshi_weights(candidate)

    raise FileNotFoundError(
        f"continual_learning teacher checkpoint is missing: {candidate!r}. "
        "로컬 경로 대신 HF repo id (예: continual_learning.hf_repo_id: "
        "nvidia/personaplex-7b-v1) 를 지정할 수도 있습니다."
    )


def _load_stage2_init_weights(model, checkpoint_dir: str) -> None:
    """Load a hybrid/full checkpoint as model-only stage-2 initialization.

    The source checkpoint supplies LoRA, VAP/backchannel, and any other
    non-face model tensors. Face tensors are always excluded so the standalone
    face checkpoint configured by face_gen.ckpt_path remains authoritative.
    Optimizer/scheduler/step state is intentionally not restored.
    """
    root = Path(checkpoint_dir)
    candidates = (
        root / "consolidated" / "lora.safetensors",
        root / "consolidated" / "consolidated.safetensors",
        root / "lora.safetensors",
        root / "consolidated.safetensors",
    )
    checkpoint_path = next((p for p in candidates if p.is_file()), None)
    if checkpoint_path is None:
        raise FileNotFoundError(
            f"No lora.safetensors or consolidated.safetensors found under "
            f"init_checkpoint_dir={checkpoint_dir!r}"
        )

    state_dict = safetensors.torch.load_file(str(checkpoint_path), device="cpu")
    source_count = len(state_dict)
    state_dict = {
        key: value
        for key, value in state_dict.items()
        if "face_module" not in key.split(".")
    }
    if not state_dict:
        raise RuntimeError(f"No non-face tensors found in init checkpoint: {checkpoint_path}")
    if not any("lora_" in key for key in state_dict):
        raise RuntimeError(
            f"Stage-2 checkpoint has no LoRA tensors: {checkpoint_path}. "
            "It is not compatible with this LoRA training configuration."
        )
    if not any("backchannel" in key.split(".") for key in state_dict):
        raise RuntimeError(
            f"Stage-2 checkpoint has no backchannel/VAP tensors: {checkpoint_path}"
        )

    incompatible = model.load_state_dict(state_dict, strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError(
            "Stage-2 checkpoint contains tensors that did not match the training "
            f"model ({len(incompatible.unexpected_keys)} unexpected): "
            f"{incompatible.unexpected_keys[:8]}"
        )
    main_logger_info(
        f"Stage-2 initialization loaded from {checkpoint_path}: "
        f"{len(state_dict)}/{source_count} non-face tensors; "
        f"{len(incompatible.missing_keys)} target tensors kept from their original "
        "initialization (base/frozen/face tensors expected)."
    )
    del state_dict


def _bernoulli_kl(student_prob: torch.Tensor, teacher_prob: torch.Tensor) -> torch.Tensor:
    eps = 1.0e-6
    student_prob = student_prob.float().clamp(eps, 1.0 - eps)
    teacher_prob = teacher_prob.float().clamp(eps, 1.0 - eps)
    return (
        teacher_prob * (teacher_prob.log() - student_prob.log())
        + (1.0 - teacher_prob)
        * ((1.0 - teacher_prob).log() - (1.0 - student_prob).log())
    )


def _selected_token_probs(
    logits: torch.Tensor,
    token_ids: tuple[int, ...],
    chunk_size: int = 256,
) -> torch.Tensor:
    """Compute a few vocabulary probabilities without materializing full fp32 softmax."""
    chunks = []
    for start in range(0, logits.shape[0], chunk_size):
        chunk = logits[start : start + chunk_size].float()
        log_z = torch.logsumexp(chunk, dim=-1, keepdim=True)
        selected = chunk[:, list(token_ids)]
        chunks.append(torch.exp(selected - log_z))
    return torch.cat(chunks, dim=0)


def _online_teacher_losses(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
    pad_id: int,
    epad_id: int,
    temperature: float,
) -> dict[str, torch.Tensor]:
    """Selective language, activity, boundary, and representation KD."""
    student_logits = student_logits[:, 0]
    teacher_logits = teacher_logits[:, 0]
    valid_mask = valid_mask.bool()
    finite = torch.isfinite(student_logits[..., 0]) & torch.isfinite(
        teacher_logits[..., 0]
    )
    valid_mask = valid_mask & finite
    zero = student_logits.new_zeros((), dtype=torch.float32)
    if not valid_mask.any():
        return {
            "text": zero,
            "speech_activity": zero,
            "turn_boundary": zero,
            "hidden": zero,
        }

    # Preserve lexical distributions only on actual word frames. Turn/silence
    # behavior is controlled separately below so AMI can still adapt it.
    word_mask = valid_mask & (targets != pad_id) & (targets != epad_id)
    word_mask &= targets >= 0
    if word_mask.any():
        student_word = student_logits[word_mask].float() / temperature
        teacher_word = teacher_logits[word_mask].float() / temperature
        teacher_prob = F.softmax(teacher_word, dim=-1)
        text_kd = F.kl_div(
            F.log_softmax(student_word, dim=-1),
            teacher_prob,
            reduction="batchmean",
        ) * (temperature * temperature)
    else:
        text_kd = zero

    # Behavioral retention: preserve the base model's frame-level decision to
    # remain PAD versus emit any non-PAD token.
    student_prob = _selected_token_probs(
        student_logits[valid_mask], (pad_id, epad_id)
    )
    teacher_prob = _selected_token_probs(
        teacher_logits[valid_mask], (pad_id, epad_id)
    )
    activity_kd = _bernoulli_kl(
        1.0 - student_prob[:, 0],
        1.0 - teacher_prob[:, 0],
    ).mean()
    boundary_kd = _bernoulli_kl(
        student_prob[:, 1],
        teacher_prob[:, 1],
    ).mean()

    hidden_mask = valid_mask
    student_h = student_hidden[hidden_mask].float()
    teacher_h = teacher_hidden[hidden_mask].float()
    hidden_kd = (1.0 - F.cosine_similarity(student_h, teacher_h, dim=-1)).mean()
    return {
        "text": text_kd,
        "speech_activity": activity_kd,
        "turn_boundary": boundary_kd,
        "hidden": hidden_kd,
    }


def train(config: str):
    """진입점: YAML 설정 파일 경로를 인수로 받습니다.

    사용법:
        torchrun --nproc_per_node=<N_GPUS> train.py config/example.yaml
    """
    args: TrainArgs = TrainArgs.load(config, drop_extra_fields=False)
    if args.text_prompt:
        raise ValueError("Moshiko gate_floor requires text_prompt to be empty")
    if args.continual_learning.enable or args.face_gen.enable:
        raise ValueError("This experiment supports frozen Moshi + ours only")
    set_logger(logging.INFO)

    with ExitStack() as exit_stack:
        _train(args, exit_stack)
    logger.info("학습 종료!")


def _train(args: TrainArgs, exit_stack: ExitStack):
    # ── 1. 초기 설정 ──────────────────────────────────────────────────────
    set_random_seed(args.seed)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # NCCL 초기화
    if "LOCAL_RANK" in os.environ:
        set_device()
        logger.info("분산 통신 초기화 중...")
        dist.init_process_group(backend=BACKEND)
    else:
        logger.error(
            "PyTorch 분산 환경이 초기화되지 않았습니다. "
            "torchrun으로 실행하세요."
        )

    # End-to-end face training must never silently degrade to LM/VAP-only
    # training. Validate every global face dependency before touching run_dir.
    if args.face_gen.enable:
        required_files = {
            "face_gen.ckpt_path": args.face_gen.ckpt_path,
            "face_gen.codec_ckpt_path": args.face_gen.codec_ckpt_path,
            "face_gen.codec_stats_path": args.face_gen.codec_stats_path,
        }
        for label, path in required_files.items():
            if not path or not os.path.isfile(path):
                raise FileNotFoundError(f"Required {label} is missing or unreadable: {path!r}")
        if not args.face_gen.flame_root or not os.path.isdir(args.face_gen.flame_root):
            raise NotADirectoryError(
                "Required face_gen.flame_root is missing or unreadable: "
                f"{args.face_gen.flame_root!r}"
            )
    if args.init_checkpoint_dir and not os.path.isdir(args.init_checkpoint_dir):
        raise NotADirectoryError(
            "init_checkpoint_dir is missing or unreadable: "
            f"{args.init_checkpoint_dir!r}"
        )

    # ── 2. run_dir 초기화 ─────────────────────────────────────────────────
    main_logger_info(f"Run dir: {args.run_dir}")
    run_dir = Path(args.run_dir)

    if is_torchrun():
        if run_dir.exists() and not args.overwrite_run_dir:
            raise RuntimeError(
                f"Run dir {run_dir} 이미 존재합니다. "
                f"`run_dir` 이름을 변경하거나 `overwrite_run_dir: true`를 설정하세요."
            )
        elif run_dir.exists():
            main_logger_info(f"Run dir 삭제 중: {run_dir}...")
            shutil.rmtree(run_dir)

    # (LoRA 모드는 args.py 및 wrapped_model.py에서 처리됨)
    pass

    dist.barrier()
    run_dir.mkdir(exist_ok=True, parents=True)

    args_path = run_dir / "args.yaml"
    if not args_path.exists():
        args.save(args_path)

    main_logger_info(f"TrainArgs:\n{pprint.pformat(dataclasses.asdict(args))}")

    # ── 3. 로거 초기화 ────────────────────────────────────────────────────
    metrics_logger: MetricsLogger = MetricsLogger(
        run_dir,
        tag="train",
        is_master=get_rank() == 0,
        wandb_args=args.wandb,
        config=dataclasses.asdict(args),
    )
    exit_stack.enter_context(logged_closing(metrics_logger, "metrics_logger"))

    eval_logger: MetricsLogger = MetricsLogger(
        run_dir,
        tag="eval",
        is_master=get_rank() == 0,
        wandb_args=args.wandb,
        config=dataclasses.asdict(args),
    )
    exit_stack.enter_context(logged_closing(eval_logger, "eval_logger"))

    # ── 4. Mimi(오디오 토크나이저) & 텍스트 토크나이저 로드 ─────────────
    main_logger_info("Mimi 및 텍스트 토크나이저 로드 중 (Moshiko)...")
    hf_repo = args.moshi_paths.hf_repo_id or DEFAULT_REPO

    mimi_path = args.moshi_paths.mimi_path
    moshi_path = args.moshi_paths.moshi_path
    tokenizer_path = args.moshi_paths.tokenizer_path

    # 로컬 경로 미설정 시 HuggingFace에서 다운로드
    if mimi_path is None or moshi_path is None or tokenizer_path is None:
        main_logger_info(
            f"경로 미설정. HF 레포에서 다운로드 중: {hf_repo}"
        )
        try:
            from huggingface_hub import hf_hub_download
            if mimi_path is None:
                mimi_path = hf_hub_download(hf_repo, MIMI_NAME)
                main_logger_info(f"  mimi → {mimi_path}")
            if moshi_path is None:
                moshi_path = hf_hub_download(hf_repo, MOSHI_NAME)
                main_logger_info(f"  moshi → {moshi_path}")
            if tokenizer_path is None:
                tokenizer_path = hf_hub_download(hf_repo, TEXT_TOKENIZER_NAME)
                main_logger_info(f"  tokenizer → {tokenizer_path}")
        except Exception as e:
            raise RuntimeError(
                f"{hf_repo} 에서 다운로드 실패: {e}\n"
                "config에 moshi_paths.mimi_path, moshi_paths.moshi_path, "
                "moshi_paths.tokenizer_path를 직접 지정해주세요."
            )

    mimi = get_mimi(mimi_path, device="cuda")
    mimi.eval()
    for p in mimi.parameters():
        p.requires_grad = False

    # ── 5. LM 설정 결정 ───────────────────────────────────────────────────
    lm_config = dict(_lm_kwargs)
    if args.moshi_paths.config_path is not None:
        import json
        with open(args.moshi_paths.config_path, "r") as f:
            lm_config = json.load(f)
        main_logger_info(f"커스텀 LM 설정 로드: {args.moshi_paths.config_path}")
    else:
        main_logger_info("Moshiko 기본 _lm_kwargs 사용 (dep_q=8)")

    # Original Moshiko predicts 8 agent codebooks; all 16 streams are inputs.
    lm_config["dep_q"] = 8

    # ── 5-1. Backchannel / VAP 설정 ──────────────────────────────────────
    if args.backchannel.enable:
        lm_config["backchannel_enabled"] = True
        lm_config["backchannel_vap_dim"] = args.backchannel.vap_dim
        lm_config["backchannel_bc_hidden"] = args.backchannel.bc_hidden
        lm_config["backchannel_gumbel_temp_init"] = args.backchannel.gumbel_temp_init
        lm_config["backchannel_gumbel_temp_min"] = args.backchannel.gumbel_temp_min
        lm_config["backchannel_gumbel_anneal_rate"] = args.backchannel.gumbel_anneal_rate
        lm_config["backchannel_fusion_trainable"] = args.backchannel.fusion_trainable
        lm_config["backchannel_fusion_bc_init"] = args.backchannel.fusion_bc_init
        lm_config["backchannel_fusion_vap_init"] = args.backchannel.fusion_vap_init
        lm_config["backchannel_fusion_vad_init"] = args.backchannel.fusion_vad_init
        lm_config["backchannel_fusion_bias_init"] = args.backchannel.fusion_bias_init
        lm_config["backchannel_fusion_evidence"] = args.backchannel.fusion_evidence
        lm_config["backchannel_fusion_calibration"] = args.backchannel.fusion_calibration
        lm_config["backchannel_fusion_calibration_init"] = args.backchannel.fusion_calibration_init
        lm_config["backchannel_decision_head"] = args.backchannel.decision_head
        lm_config["backchannel_decision_hidden"] = args.backchannel.decision_hidden
        lm_config["backchannel_aux_to_trunk"] = args.backchannel.aux_to_trunk
        lm_config["backchannel_fusion_gate_frames"] = args.backchannel.fusion_gate_frames
        if args.backchannel.pad_token_id is not None:
            lm_config["backchannel_pad_token_id"] = args.backchannel.pad_token_id
        if args.backchannel.epad_token_id is not None:
            lm_config["backchannel_epad_token_id"] = args.backchannel.epad_token_id

        if args.backchannel.module_type == "mlp":
            main_logger_info("Backchannel MLP VAP 모듈 활성화 (LM 내부)")
            lm_config["backchannel_module_type"] = "mlp"
        elif args.backchannel.module_type == "vap_gpt":
            main_logger_info("VapGPT Backchannel 모듈 활성화 (LM 내부, z_s 입력)")
            lm_config["backchannel_module_type"] = "vap_gpt"
            lm_config["backchannel_vap_repo_path"] = args.backchannel.vap_gpt_repo_path
            lm_config["backchannel_vap_checkpoint"] = args.backchannel.vap_gpt_checkpoint
            lm_config["backchannel_vap_channel_layers"] = args.backchannel.vap_gpt_channel_layers
            lm_config["backchannel_vap_cross_layers"] = args.backchannel.vap_gpt_cross_layers
            lm_config["backchannel_vap_num_heads"] = args.backchannel.vap_gpt_num_heads
            lm_config["backchannel_vap_dropout"] = args.backchannel.vap_gpt_dropout
            lm_config["backchannel_vap_use_silence_ctx_proj"] = getattr(args.backchannel, "use_silence_ctx_proj", True)
            main_logger_info(
                f"VapGPTBackchannelModule 설정 완료 "
                f"(checkpoint={args.backchannel.vap_gpt_checkpoint})"
            )
        else:
            raise ValueError(
                f"알 수 없는 backchannel.module_type: '{args.backchannel.module_type}'. "
                "지원 값: 'mlp', 'vap_gpt'"
            )

    # ── 5-2. Face Generation 설정 ─────────────────────────────────────────
    if args.face_gen.enable:
        face_dir = os.path.normpath(os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../moshi/moshi/models/face",
        ))
        lm_config["face_module_enabled"] = True
        lm_config["face_module_dir"] = face_dir
        lm_config["face_module_checkpoint"] = args.face_gen.ckpt_path
        lm_config["face_module_version"] = int(args.face_gen.model_version)
        lm_config["face_module_hidden_dim"] = args.face_gen.hidden_dim
        lm_config["face_module_layers"] = args.face_gen.layers
        lm_config["face_module_heads"] = args.face_gen.heads
        lm_config["face_module_code_dim"] = args.face_gen.code_dim
        lm_config["face_module_prior_warmup_frames"] = args.face_gen.prior_warmup_frames
        lm_config["face_module_ss_prob"] = args.face_gen.scheduled_sampling_prob
        lm_config["face_module_ss_ramp_steps"] = args.face_gen.scheduled_sampling_ramp_steps
        lm_config["face_module_ss_start_step"] = args.face_gen.warmup_steps
        lm_config["face_module_ss_keep_head_frames"] = args.face_gen.scheduled_sampling_keep_head_frames
        lm_config["face_module_detach_llm_features"] = args.face_gen.detach_llm_features
        main_logger_info(
            f"Face generation module 활성화 "
            f"(checkpoint={args.face_gen.ckpt_path})"
        )

    # ── 6. LM 모델 로드 및 FSDP 샤딩 ─────────────────────────────────────
    main_logger_info("Moshiko LM + ours 모듈 로드 중...")
    model = get_fsdp_model(args, moshi_path, lm_config)
    if args.init_checkpoint_dir:
        _load_stage2_init_weights(model, args.init_checkpoint_dir)

    # Frozen original PersonaPlex reference. Its special forward skips audio
    # Depformer, VAP, and face computation; only text logits and backbone hidden
    # states needed for continual-learning losses are retained.
    teacher_model = None
    if args.continual_learning.enable:
        teacher_checkpoint = _resolve_teacher_checkpoint(
            args.continual_learning, moshi_path
        )
        main_logger_info(
            f"Frozen online PersonaPlex teacher loading: {teacher_checkpoint}"
        )
        teacher_model = get_moshi_lm(
            teacher_checkpoint,
            device="cuda",
            dtype=getattr(torch, args.param_dtype),
        )
        teacher_model.eval()
        for parameter in teacher_model.parameters():
            parameter.requires_grad = False
        main_logger_info(
            "Online teacher ready (text/hidden-only forward; all parameters frozen)."
        )

    # ── 6-1. ARTalkCodec (VAE) 로드 (face_gen 활성 시) ────────────────────
    # The codec is frozen and used only to compute z_target = quant_to_sum_feat(gt_face_motion).
    # It is NOT wrapped in FSDP — each rank holds a full copy (codec is small, ~50M params).
    face_codec = None
    if args.face_gen.enable and args.face_gen.codec_ckpt_path:
        from types import SimpleNamespace
        face_dir = os.path.normpath(os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../moshi/moshi/models/face",
        ))
        if face_dir not in sys.path:
            sys.path.insert(0, face_dir)
        from core.models.artalk_codec import ARTalkCodec  # noqa: PLC0415
        codec_cfg = SimpleNamespace(
            MOTION_DIM=54,
            V_CODE_DIM=args.face_gen.code_dim,
            V_PATCH_NUMS=[1, 10, 20, 50, 100],
            T_DEPTH=args.face_gen.codec_depth,
            T_NUM_HEADS=args.face_gen.codec_heads,
            T_HIDDEN_DIM=args.face_gen.codec_hidden_dim,
            STATS_PATH=args.face_gen.codec_stats_path or "",
        )
        face_codec = ARTalkCodec(codec_cfg, init_submodule=False)
        _codec_ckpt = torch.load(args.face_gen.codec_ckpt_path, map_location="cpu", weights_only=True)
        face_codec.load_state_dict(_codec_ckpt["model"], strict=True)
        face_codec.eval().cuda()
        for p in face_codec.parameters():
            p.requires_grad = False
        main_logger_info(f"ARTalkCodec loaded from: {args.face_gen.codec_ckpt_path}")

    # ── 7. 텍스트 토크나이저 & Interleaver 구성 ───────────────────────────
    main_logger_info("텍스트 토크나이저 로드 중...")
    import sentencepiece
    spm = sentencepiece.SentencePieceProcessor()
    spm.Load(tokenizer_path)

    # ── 텍스트 프롬프트 사전 토크나이즈 ──────────────────────────────────
    # inference의 lm_gen.text_prompt_tokens = spm.encode(text_prompt) 와 동일.
    prompt_ids: list[int] = spm.encode(args.text_prompt) if args.text_prompt else []
    T_p: int = len(prompt_ids)
    if T_p:
        main_logger_info(
            f"텍스트 프롬프트 활성화: {T_p}토큰 prefix로 삽입 "
            f"(inference step_system_prompts 정렬)\n  프롬프트: {args.text_prompt!r}"
        )

    if args.backchannel.enable and args.backchannel.bc_event_loss_weight > 0:
        _bc = args.backchannel
        if _bc.bc_target_mode == "vap":
            _horizon_ms = sum([200, 400, 600, 800][: _bc.bc_vap_horizon_bins])
            main_logger_info(
                f"BC head target: 'vap' — agent가 향후 {_horizon_ms}ms "
                f"({_bc.bc_vap_horizon_bins} bin) 내 발화 시작 "
                f"{'+ 같은 구간 유저 침묵 ' if _bc.bc_vap_require_user_silent else ''}"
                f"→ class 1. VAP manifest 필요."
            )
        else:
            main_logger_info("BC head target: 'epad' — GT 텍스트의 [EPAD] 토큰 위치 → class 1.")

    interleaver = Interleaver(
        spm,
        mimi.frame_rate,
        model.text_padding_token_id,
        model.end_of_text_padding_id,
        model.zero_token_id,
        keep_main_only=True,
    )
    interleaved_tokenizer = InterleavedTokenizer(
        mimi, interleaver, duration_sec=args.duration_sec,
        vap_manifest_path=args.data.vap_manifest,
        flame_root=args.face_gen.flame_root if args.face_gen.enable else "",
        flame_speaker=args.face_gen.flame_speaker,
    )

    # ── 8. 데이터 로더 구성 ───────────────────────────────────────────────
    data_loader = build_data_loader(
        instruct_tokenizer=interleaved_tokenizer,
        args=args.data,
        batch_size=args.batch_size,
        seed=args.seed,
        rank=get_rank(),
        world_size=get_world_size(),
        is_eval=False,
    )

    # eval_data_loader은 매 평가 시마다 새로 생성합니다.
    # is_eval=True의 유한 이터레이터를 재사용하면 백그라운드 스레드가 데이터셋을
    # 소진한 후 종료되고, 이후 _q.get()이 영구 블록됩니다.

    # ── 9. dtype 설정 ─────────────────────────────────────────────────────
    param_dtype = getattr(torch, args.param_dtype)

    # ── 10. 옵티마이저 & 스케줄러 ─────────────────────────────────────────
    base_params = []
    fusion_params = []
    decision_params = []
    face_core_params = []
    face_llm_proj_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "backchannel_fusion_" in name:
            fusion_params.append(param)
        elif "backchannel.decision_head" in name:
            decision_params.append(param)
        elif "face_module" not in name:
            base_params.append(param)
        elif "llm_proj" in name:
            face_llm_proj_params.append(param)
        else:
            face_core_params.append(param)

    param_groups = [{"params": base_params, "lr": args.optim.lr, "group_name": "base"}]
    max_lrs = [args.optim.lr]
    if args.backchannel.enable and args.backchannel.fusion_trainable:
        if not fusion_params:
            raise RuntimeError(
                "fusion_trainable=True but no trainable backchannel fusion parameters were found"
            )
        param_groups.append({
            "params": fusion_params,
            "lr": args.backchannel.fusion_lr,
            "weight_decay": 0.0,
            "group_name": "fusion",
        })
        max_lrs.append(args.backchannel.fusion_lr)
        main_logger_info(
            "Trainable START-vs-rest fusion: "
            f"lr={args.backchannel.fusion_lr:.2e}, "
            f"bc={args.backchannel.fusion_bc_init:.3f}, "
            f"vap={args.backchannel.fusion_vap_init:.3f}, "
            f"vad={args.backchannel.fusion_vad_init:.3f}, "
            f"bias={args.backchannel.fusion_bias_init:.3f}"
        )
    if decision_params:
        decision_lr = args.backchannel.decision_lr or args.optim.lr
        param_groups.append({"params": decision_params, "lr": decision_lr, "group_name": "decision"})
        max_lrs.append(decision_lr)
        main_logger_info(f"Decision head: lr={decision_lr:.2e}, aux_to_trunk={args.backchannel.aux_to_trunk}")
    if args.face_gen.enable:
        face_core_lr = args.face_gen.core_lr or args.optim.lr
        face_llm_proj_lr = args.face_gen.llm_proj_lr or args.optim.lr
        if not face_core_params:
            raise RuntimeError("face_gen is enabled but no trainable face core parameters were found")
        if not face_llm_proj_params:
            raise RuntimeError("face_gen is enabled but no trainable face llm_proj parameters were found")
        param_groups.extend([
            {"params": face_core_params, "lr": face_core_lr, "group_name": "face_core"},
            {"params": face_llm_proj_params, "lr": face_llm_proj_lr, "group_name": "face_llm_proj"},
        ])
        max_lrs.extend([face_core_lr, face_llm_proj_lr])
        main_logger_info(
            "Optimizer learning rates: "
            f"base={args.optim.lr:.2e}, face_core={face_core_lr:.2e}, "
            f"face_llm_proj={face_llm_proj_lr:.2e}"
        )
    optimizer = AdamW(
        param_groups,
        betas=(0.9, 0.95),
        eps=1e-08,
        weight_decay=args.optim.weight_decay,
    )

    scheduler = lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lrs,
        total_steps=args.max_steps,
        pct_start=args.optim.pct_start,
    )

    state = TrainState(args.max_steps)

    # ── 11. 체크포인터 초기화 ─────────────────────────────────────────────
    if args.do_ckpt:
        checkpointer = Checkpointer(
            model=model,
            state=state,
            config=lm_config,
            run_dir=run_dir,
            optimizer=optimizer,
            num_ckpt_keep=args.num_ckpt_keep,
            full_finetuning=args.full_finetuning,
            keep_best_metric=args.ckpt_keep_best_metric,
            keep_best_n=args.ckpt_keep_best_n,
            save_backchannel_only=args.freeze_backbone,
        )

    # ── 12. 학습 준비 ──────────────────────────────────────────────────────
    # FSDP MixedPrecision이 bf16 변환을 내부적으로 처리합니다.
    # prepare_mixed_precision은 불필요(fp32 복사본을 별도 유지하여 메모리 낭비)
    scaler = None  # bf16 사용 시 GradScaler 불필요

    # ── 13. 학습 루프 ─────────────────────────────────────────────────────
    model.train()
    torch.cuda.empty_cache()
    main_logger_info("학습 시작!")

    # ── Logit-Adjustment prior for the 3-class BC head (Menon et al., ICLR 2021) ──
    # Running class counts over valid BC targets (PAD/EPAD/WORD), Laplace-smoothed
    # (init=1 per class). log π is ADDED to the logits inside the training CE, so
    # the network's raw logits approximate prior-free scores and plain argmax at
    # inference is the intended (τ-tempered balanced) rule — no post-hoc correction.
    bc_class_counts = torch.ones(3, device="cuda", dtype=torch.float64)
    fusion_monitor = None
    if args.fusion_dynamics.enable:
        fusion_monitor = FusionDynamicsMonitor(model, optimizer, args, run_dir)
        model.fusion_dynamics_diagnostics = True
        model.fusion_dynamics_prompt_length = T_p

    while state.step < args.max_steps:
        state.start_step()
        is_last_step = state.step == args.max_steps
        dynamics_stop = False
        aux_coverage = {key: 0 for key in ("vap", "vad", "bc")}
        aux_mode = {key: "missing_labels" for key in ("vap", "vad", "bc")}

        loss = torch.tensor([0.0], device="cuda")
        vap_loss_val = torch.tensor([0.0], device="cuda")
        # commitment_loss_val = torch.tensor([0.0], device="cuda")
        face_loss_val = torch.tensor([0.0], device="cuda")
        face_component_vals = {
            name: torch.tensor([0.0], device="cuda")
            for name in (
                "motion_loss", "prior_loss", "z_mse_loss", "z_bce_loss",
                "jaw_loss", "velocity_loss", "regularization_loss", "gate_loss",
                "pred_jaw_abs", "gt_jaw_abs", "jaw_amplitude_ratio",
                "pred_jaw_velocity_abs", "gt_jaw_velocity_abs",
                "prior_jaw_abs", "delta_jaw_abs", "residual_jaw_abs", "gate_jaw",
                "scheduled_sampling_prob",
            )
        }
        bc_event_loss_val = torch.tensor([0.0], device="cuda")
        silence_loss_val = torch.tensor([0.0], device="cuda")
        turn_end_floor_val = torch.tensor([0.0], device="cuda")
        turn_end_frac_val = torch.tensor([0.0], device="cuda")
        kd_vals = {
            name: torch.tensor([0.0], device="cuda")
            for name in ("text", "speech_activity", "turn_boundary", "hidden")
        }
        bc_stats_accum: dict | None = None
        epad_counts = torch.zeros(4, device="cuda")  # [tp, fp, fn, tn] for [EPAD] prediction
        bc_head_counts = torch.zeros(4, device="cuda")  # [tp, fp, fn, tn] for bc_logits class 1
        n_batch_tokens: int = 0
        n_real_tokens: int = 0
        face_loss_skipped_no_data: int = 0   # batches where face_motion_gt was None (missing FLAME files)
        face_loss_skipped_nonfinite: int = 0 # batches where face_loss was NaN/Inf
        face_warmup_active = (
            args.face_gen.enable
            and args.face_gen.warmup_steps > 0
            and state.step <= args.face_gen.warmup_steps
        )

        for i in range(args.num_microbatches):
            batch = next(data_loader)
            codes = batch.codes  # [B, K, T]

            # ── 텍스트 프롬프트 prefix 삽입 ───────────────────────────────
            # inference: step_system_prompts() → _step_text_prompt_core()에서
            # 토큰마다 text_token=prompt_id, audio=silence 로 step()을 호출하는 것과 동일.
            # 학습: codes의 앞에 [prompt_ids ; zero_token_id×(K-1)] 블록을 붙임.
            # 손실은 prefix 구간을 제외하고 원본 codes에 대해서만 계산.
            if T_p:
                prefix = torch.full(
                    [codes.shape[0], codes.shape[1], T_p],
                    model.zero_token_id,
                    device=codes.device, dtype=codes.dtype,
                )
                prefix[:, 0, :] = torch.tensor(prompt_ids, device=codes.device)
                codes_in = torch.cat([prefix, codes], dim=2)  # [B, K, T_p+T]
            else:
                codes_in = codes

            # 마지막 microbatch 전까지는 FSDP all-reduce 억제 (단일 GPU는 no_sync 없으므로 스킵)
            sync_ctx = (
                contextlib.nullcontext()
                if (i == args.num_microbatches - 1 or not hasattr(model, "no_sync"))
                else model.no_sync()
            )
            with sync_ctx:
                # FSDP의 forward pre-hook이 정상 발동하도록 __call__ -> forward 호출
                voice_prompt_embs = batch.voice_prompt_embs
                if voice_prompt_embs is not None:
                    voice_prompt_embs = voice_prompt_embs.to(codes.device, non_blocking=True)  # int64 mimi codes

                # ── Face generation inputs ────────────────────────────────
                audio_feat = None
                gt_face_motion = None
                mimi_for_model = None
                if args.face_gen.enable:
                    if args.face_gen.use_generated_audio_feat:
                        # audio_feat will be derived inside the model from its own predicted
                        # audio codes (argmax of depformer logits). Pass mimi so the model
                        # can call mimi.decode_latent internally.
                        mimi_for_model = mimi
                    else:
                        # Teacher-forced: decode GT audio codes to Mimi latent features.
                        # Uses frozen mimi (no grad) — safe to call outside the model.
                        with torch.no_grad():
                            # codes[:, 1:9] = first 8 agent audio codebooks (Mimi's n_q).
                            # Clamp to min=0 so zero_token_id (-1) padding positions don't
                            # produce out-of-bounds indices inside F.embedding in the VQ decoder.
                            audio_codes = codes[:, 1:9].clamp(min=0)
                            audio_feat = mimi.decode_latent(audio_codes).transpose(1, 2)  # [B, T, 512]
                            audio_feat = audio_feat.to(dtype=param_dtype)
                            if T_p:
                                # Prepend T_p silence frames so audio_feat aligns with codes_in.
                                zero_feat = torch.zeros(
                                    codes.shape[0], T_p, audio_feat.shape[2],
                                    device=audio_feat.device, dtype=audio_feat.dtype,
                                )
                                audio_feat = torch.cat([zero_feat, audio_feat], dim=1)  # [B, T_p+T, 512]
                    if batch.face_motion_gt is not None:
                        gt_face_motion = batch.face_motion_gt.to(codes.device, dtype=param_dtype)
                        if T_p:
                            # Face runs at 25fps = 2× mimi 12.5fps → T_p mimi frames = 2*T_p face frames.
                            T_face_p = T_p * 2
                            zero_motion = torch.zeros(
                                codes.shape[0], T_face_p, gt_face_motion.shape[2],
                                device=gt_face_motion.device, dtype=gt_face_motion.dtype,
                            )
                            gt_face_motion = torch.cat([zero_motion, gt_face_motion], dim=1)
                    else:
                        # batch.face_motion_gt is None when ≥1 sample in the batch has no
                        # matching FLAME .npy file — the collate_fn uses all(...) so one
                        # missing file silently drops face loss for the entire batch.
                        face_loss_skipped_no_data += 1

                # ── Per-speaker Mimi features for VapGPT backchannel ─────────────────
                bc_audio_feats = None
                if (args.backchannel.enable
                        and args.backchannel.module_type == "vap_gpt"):
                    with torch.no_grad():
                        _agent = mimi.decode_latent(codes[:, 1:9].clamp(min=0)).transpose(1, 2)
                        _user  = mimi.decode_latent(codes[:, 9:17].clamp(min=0)).transpose(1, 2)
                        _agent = _agent.to(dtype=param_dtype)
                        _user  = _user.to(dtype=param_dtype)
                        if T_p:
                            _zero = torch.zeros(codes.shape[0], T_p, _agent.shape[2],
                                                device=_agent.device, dtype=_agent.dtype)
                            _agent = torch.cat([_zero, _agent], dim=1)
                            _user  = torch.cat([_zero, _user],  dim=1)
                    bc_audio_feats = (_agent, _user)

                teacher_text_logits = None
                teacher_text_mask = None
                teacher_hidden = None
                if teacher_model is not None:
                    with torch.no_grad():
                        (
                            teacher_text_logits,
                            teacher_text_mask,
                            teacher_hidden,
                        ) = teacher_model.forward_teacher(
                            codes_in,
                            voice_prompt_embs=voice_prompt_embs,
                        )

                output = model(codes_in, step=state.step, voice_prompt_embs=voice_prompt_embs,
                               audio_feat=audio_feat, gt_face_motion=gt_face_motion,
                               mimi=mimi_for_model, bc_audio_feats=bc_audio_feats)

                # Text at padded positions is already zero_token_id (-1) → text_mask=False
                # regardless, so no extra masking is needed there.
                # Slice off the T_p prompt-prefix frames — loss is on the conversation only.
                text_mask = output.text_mask[:, :, T_p:]
                audio_mask = output.mask[:, :, T_p:]

                # Exclude the zero-padded tail of short clips/windows (batch.valid_mask
                # is False past the real audio length) from the audio loss. Without this,
                # every batch's last window contributes an artificial "predict silence"
                # signal proportional to how much of it is padding, on top of any real
                # silence in the content — biasing the model toward under-talking.
                if batch.valid_mask is not None:
                    audio_mask = audio_mask & batch.valid_mask.to(audio_mask.device).unsqueeze(1)

                text_loss = compute_loss_with_mask(
                    output.text_logits[:, :, T_p:],
                    codes[:, : model.audio_offset],
                    text_mask,
                    mode="text",
                    text_padding_weight=args.text_padding_weight,
                    text_padding_ids={model.text_padding_token_id},
                    epad_ids={model.end_of_text_padding_id},
                    epad_weight=args.epad_weight,
                )
                audio_loss = compute_loss_with_mask(
                    output.logits[:, :, T_p:],
                    codes[:, model.audio_offset : model.audio_offset + model.dep_q],
                    audio_mask,
                    mode="audio",
                    first_codebook_weight_multiplier=args.first_codebook_weight_multiplier,
                )

                # [EPAD] 예측 정확도(acc/precision/recall/f1) 집계용 confusion counts
                epad_counts += epad_confusion_counts(
                    output.text_logits[:, :, T_p:],
                    codes[:, : model.audio_offset],
                    text_mask,
                    model.end_of_text_padding_id,
                )

                mb_loss = text_loss + audio_loss

                # ── Frozen PersonaPlex online-teacher distillation ────────────────
                if teacher_model is not None:
                    if output.transformer_out is None:
                        raise RuntimeError(
                            "Student LMOutput is missing transformer_out required "
                            "for continual-learning hidden distillation."
                        )
                    kd_mask = text_mask[:, 0].clone()
                    kd_mask &= teacher_text_mask[:, 0, T_p:]
                    if batch.valid_mask is not None:
                        kd_mask &= batch.valid_mask.to(kd_mask.device)
                    kd_losses = _online_teacher_losses(
                        student_logits=output.text_logits[:, :, T_p:],
                        teacher_logits=teacher_text_logits[:, :, T_p:],
                        student_hidden=output.transformer_out[:, T_p:],
                        teacher_hidden=teacher_hidden[:, T_p:],
                        targets=codes[:, 0],
                        valid_mask=kd_mask,
                        pad_id=model.text_padding_token_id,
                        epad_id=model.end_of_text_padding_id,
                        temperature=args.continual_learning.temperature,
                    )
                    cl = args.continual_learning
                    mb_loss = (
                        mb_loss
                        + cl.text_kd_weight * kd_losses["text"]
                        + cl.speech_activity_kd_weight
                        * kd_losses["speech_activity"]
                        + cl.turn_boundary_kd_weight
                        * kd_losses["turn_boundary"]
                        + cl.hidden_kd_weight * kd_losses["hidden"]
                    )
                    for name, value in kd_losses.items():
                        kd_vals[name] += value.detach()

                # ── VAP 보조 손실 ─────────────────────────────────────────
                vap_loss = None
                if args.backchannel.enable:
                    # Slice off prompt prefix so VAP logits align with batch.vap_targets [B, T].
                    vap_logits = output.vap_logits[:, T_p:] if output.vap_logits is not None else None

                    if vap_logits is not None:
                        vap_targets_tensor = batch.vap_targets  # [B, T]
                        # Mask silence-padded frames: set to ignore_index so cross_entropy skips them
                        if vap_targets_tensor is not None and batch.valid_mask is not None:
                            vap_targets_tensor = vap_targets_tensor.masked_fill(
                                ~batch.valid_mask.to(vap_targets_tensor.device), -100
                            )
                        if vap_targets_tensor is not None:
                            flat_targets = vap_targets_tensor.view(-1).long()
                            n_valid = (flat_targets != -100).sum().item()
                            if fusion_monitor is not None:
                                aux_coverage["vap"] += n_valid
                                aux_mode["vap"] = "labels"
                            if n_valid > 0:
                                num_vap_classes = vap_logits.shape[-1]
                                flat_targets = flat_targets.clamp(-100, num_vap_classes - 1)
                                vap_loss = F.cross_entropy(
                                    vap_logits.reshape(-1, num_vap_classes),
                                    flat_targets,
                                    ignore_index=-100,
                                )
                                if not torch.isnan(vap_loss):
                                    mb_loss = mb_loss + args.backchannel.vap_loss_weight * vap_loss
                                    vap_loss_val += vap_loss.detach()
                            # else: no valid VAP targets in this batch — skip
                        else:
                            # Fallback: entropy regularization when no manifest targets
                            vap_probs = F.softmax(vap_logits, dim=-1)
                            vap_entropy = -(vap_probs * (vap_probs + 1e-8).log()).sum(dim=-1).mean()
                            vap_loss = -vap_entropy
                            if fusion_monitor is not None:
                                aux_mode["vap"] = "entropy_fallback"
                            mb_loss = mb_loss + args.backchannel.vap_loss_weight * vap_loss
                            vap_loss_val += vap_loss.detach()

                    # Alt 3: Commitment loss — bc_mlp + silence_gate alignment with depformer
                    # if output.commitment_loss is not None and not torch.isnan(output.commitment_loss):
                    #     mb_loss = mb_loss + args.backchannel.commitment_loss_weight * output.commitment_loss
                    #     commitment_loss_val += output.commitment_loss.detach()

                # ── Direct BC supervision (3-class CE) ────────────────────────────────
                # bc_logits [B, T, 3] predicts 0=PAD / 1=EPAD (backchannel slot) / 2=WORD.
                # backchannel.bc_target_mode selects what counts as a backchannel:
                #   "epad" — GT text tokens: responses the dataset agent actually made
                #   "vap"  — VAP manifest turn-taking: slots where a response would fit
                # Both share this head, the ±K boundary-ignore band, and the
                # Logit-Adjustment CE below. See build_bc_targets() for label details.
                if (args.backchannel.enable
                        and (args.backchannel.bc_event_loss_weight > 0 or fusion_monitor is not None)
                        and output.bc_logits is not None):
                    cls_tgt = build_bc_targets(
                        mode=args.backchannel.bc_target_mode,
                        text_tokens=codes[:, 0],  # [B, T] GT text (undelayed, content only)
                        pad_id=model.text_padding_token_id,
                        epad_id=model.end_of_text_padding_id,
                        onset_ignore_frames=args.backchannel.bc_onset_ignore_frames,
                        valid_mask=batch.valid_mask,
                        vap_targets=batch.vap_targets,
                        vad_targets=batch.vad_targets,
                        vap_horizon_bins=args.backchannel.bc_vap_horizon_bins,
                        vap_require_user_silent=args.backchannel.bc_vap_require_user_silent,
                    )
                    bc_logits_c = output.bc_logits[:, T_p:]  # [B, T, 3] — strip prompt prefix
                    if (cls_tgt != -100).any():
                        if fusion_monitor is not None:
                            aux_coverage["bc"] += int((cls_tgt != -100).sum().item())
                            aux_mode["bc"] = "labels"
                        # Logit-Adjustment loss (Menon et al., ICLR 2021): CE on
                        # (logits + τ·log π) instead of class-weighted CE. Unlike
                        # pos_weight (which multiplies rare-class gradients and
                        # destabilises training), LA shifts the decision margin
                        # additively while keeping gradient scales balanced.
                        valid_t = cls_tgt[cls_tgt != -100]
                        bc_class_counts += torch.bincount(valid_t, minlength=3).to(bc_class_counts)
                        log_prior = torch.log(bc_class_counts / bc_class_counts.sum()).to(
                            device=codes.device, dtype=torch.float32,
                        )
                        bc_logits_adj = bc_logits_c.float() + args.backchannel.bc_la_tau * log_prior
                        bc_event_loss = F.cross_entropy(
                            bc_logits_adj.reshape(-1, 3),
                            cls_tgt.reshape(-1),
                            ignore_index=-100,
                        )
                        mb_loss = mb_loss + args.backchannel.bc_event_loss_weight * bc_event_loss
                        bc_event_loss_val += bc_event_loss.detach()
                        # BC-head quality against its own target — the only metric
                        # that reflects bc_target_mode (epad_f1 scores the text head).
                        bc_head_counts += bc_head_confusion_counts(
                            bc_logits_c.detach(), cls_tgt
                        )

                # ── Current-frame VAD supervision (v2: BCE, both streams) ─────────────
                # vad_logits [B, T, 2] predicts CURRENT-frame voice activity
                # ([...,0]=user, [...,1]=agent); targets come straight from the VAP
                # manifest's cur_va_int (row0=user, row1=agent, -100 where the manifest
                # has no entry for that frame).
                if (args.backchannel.enable
                        and (args.backchannel.vad_loss_weight > 0 or fusion_monitor is not None)
                        and output.vad_logits is not None
                        and batch.vad_targets is not None):
                    vad_t = batch.vad_targets.to(codes.device)         # [B, 2, T]
                    vad_l = output.vad_logits[:, T_p:].transpose(1, 2)  # [B, 2, T] — strip prefix
                    vmask = vad_t >= 0  # drop frames with no manifest VAD entry
                    if batch.valid_mask is not None:
                        vmask = vmask & batch.valid_mask.to(codes.device).unsqueeze(1).expand_as(vad_t)
                    if vmask.any():
                        if fusion_monitor is not None:
                            aux_coverage["vad"] += int(vmask.sum().item())
                            aux_mode["vad"] = "labels"
                        vad_loss = F.binary_cross_entropy_with_logits(
                            vad_l[vmask].float(), vad_t[vmask].float(),
                        )
                        mb_loss = mb_loss + args.backchannel.vad_loss_weight * vad_loss
                        silence_loss_val += vad_loss.detach()  # reuse tracker; logged as vad_loss

                # ── Turn-end floor: the listener may raise but not lower EPAD right after
                # the user stops (agent silent), leaving the backbone's turn starts intact.
                if (args.backchannel.enable and args.backchannel.turn_end_floor_weight > 0
                        and output.fusion_residual is not None and batch.vad_targets is not None):
                    # agent silent = no non-PAD text token in the previous gate window (6 if no gate)
                    agent_speaking = model.fusion_speaking_mask(
                        codes[:, 0], args.backchannel.fusion_gate_frames or 6)
                    te_mask = turn_end_mask(
                        batch.vad_targets.to(codes.device)[:, 0], agent_speaking,
                        args.backchannel.turn_end_window,
                        batch.valid_mask.to(codes.device) if batch.valid_mask is not None else None,
                    )
                    if te_mask.any():
                        r_conv = output.fusion_residual[:, T_p:].float()
                        floor_loss = torch.relu(-r_conv[te_mask]).mean()
                        mb_loss = mb_loss + args.backchannel.turn_end_floor_weight * floor_loss
                        turn_end_floor_val += floor_loss.detach()
                    turn_end_frac_val += te_mask.float().mean().detach()

                # ── Face motion reconstruction loss (full reference loss) ──
                face_loss = None
                if args.face_gen.enable:
                    if output.face_outputs is None:
                        raise RuntimeError(
                            f"[step {state.step}] face_gen is enabled but face_outputs is None "
                            f"(gt_face_motion={'ok' if gt_face_motion is not None else 'MISSING'}, "
                            f"face_codec={'ok' if face_codec is not None else 'MISSING'})."
                        )
                    elif gt_face_motion is not None and face_codec is not None:
                        # Build [B, T_face] bool mask.
                        # When T_p > 0, gt_face_motion was prepended with 2*T_p zero frames
                        # (face runs at 25fps = 2× mimi 12.5fps) — exclude those from loss.
                        T_face_p = T_p * 2
                        valid_face_mask = None
                        if T_p or batch.valid_face_frames is not None:
                            T_face = gt_face_motion.shape[1]
                            t_idx = torch.arange(T_face, device=codes.device).unsqueeze(0)  # [1, T_face]
                            if batch.valid_face_frames is not None:
                                valid_face_mask = (t_idx >= T_face_p) & (
                                    t_idx < T_face_p + batch.valid_face_frames.to(codes.device).unsqueeze(1)
                                )
                            else:
                                valid_face_mask = t_idx >= T_face_p  # [1, T_face], broadcasts to [B, T_face]
                        if valid_face_mask is not None and not valid_face_mask.any() and get_rank() == 0:
                            logger.warning(
                                f"[step {state.step}] valid_face_mask is all-False — "
                                f"valid_face_frames={batch.valid_face_frames}, "
                                f"T_face_p={T_face_p}, T_face={gt_face_motion.shape[1]}. "
                                f"Face loss will be 0."
                            )
                        face_loss, face_components = compute_face_loss(
                            output.face_outputs, gt_face_motion, face_codec, args.face_gen,
                            valid_face_mask=valid_face_mask,
                            return_components=True,
                        )
                        face_components["scheduled_sampling_prob"] = output.face_outputs[
                            "scheduled_sampling_prob"
                        ]
                        if torch.isfinite(face_loss):
                            if face_loss.item() == 0.0 and get_rank() == 0:
                                logger.warning(
                                    f"[step {state.step}] face_loss computed as exactly 0.0 "
                                    f"(mask valid={valid_face_mask.sum().item() if valid_face_mask is not None else 'N/A'} frames). "
                                    f"Check FLAME data quality."
                                )
                            # During face warm-up, keep forward/loss monitoring active but
                            # exclude face loss from the objective. This blocks its gradient
                            # from both the face module and the upstream LM features.
                            if not face_warmup_active:
                                mb_loss = mb_loss + args.face_gen.face_loss_weight * face_loss
                            face_loss_val += face_loss.detach()
                            for name, component in face_components.items():
                                face_component_vals[name] += component.detach()
                        else:
                            face_loss_skipped_nonfinite += 1
                            logger.warning(f"[step {state.step}] Non-finite face_loss={face_loss.item():.4f}, skipping.")

                n_batch_tokens += text_mask.numel() + audio_mask.numel()
                n_real_tokens += (
                    torch.sum(text_mask).item() + torch.sum(audio_mask).item()
                )

                if fusion_monitor is not None and not torch.isfinite(mb_loss).all():
                    fusion_monitor.fail("failed_nonfinite", f"Nonfinite loss at step {state.step}")
                mb_loss.backward()

            loss += mb_loss.detach()
            if args.backchannel.enable and output.bc_stats is not None:
                if bc_stats_accum is None:
                    bc_stats_accum = {k: v.clone() for k, v in output.bc_stats.items()}
                else:
                    for k in bc_stats_accum:
                        bc_stats_accum[k] = bc_stats_accum[k] + output.bc_stats[k]
            del output, mb_loss, text_loss, audio_loss
            if args.backchannel.enable and vap_loss is not None:
                del vap_loss
            if bc_audio_feats is not None:
                del bc_audio_feats
            if args.face_gen.enable:
                del audio_feat, gt_face_motion, face_loss

        if args.num_microbatches > 1:
            loss /= args.num_microbatches
            if args.backchannel.enable:
                vap_loss_val /= args.num_microbatches
                # commitment_loss_val /= args.num_microbatches
                bc_event_loss_val /= args.num_microbatches
                silence_loss_val /= args.num_microbatches
                turn_end_floor_val /= args.num_microbatches
                turn_end_frac_val /= args.num_microbatches
                if bc_stats_accum is not None:
                    for k in bc_stats_accum:
                        bc_stats_accum[k] = bc_stats_accum[k] / args.num_microbatches
            if args.face_gen.enable:
                face_loss_val /= args.num_microbatches
                for name in face_component_vals:
                    face_component_vals[name] /= args.num_microbatches
            if teacher_model is not None:
                for name in kd_vals:
                    kd_vals[name] /= args.num_microbatches
            for p in model.parameters():
                if p.requires_grad and p.grad is not None:
                    p.grad.div_(args.num_microbatches)

        # 그래디언트 클리핑
        if fusion_monitor is not None:
            fusion_monitor.before_clip()
        torch.nn.utils.clip_grad_norm_(list(model.parameters()), args.max_norm)

        if args.face_gen.enable and args.face_gen.warmup_steps > 0:
            if state.step == 1 and get_rank() == 0:
                logger.info(
                    f"Face loss warm-up active for steps 1-{args.face_gen.warmup_steps}: "
                    "face loss is monitored but excluded from backpropagation."
                )
            if state.step == args.face_gen.warmup_steps and get_rank() == 0:
                logger.info(
                    f"[step {state.step}] Face loss warm-up complete; "
                    f"face gradients will be enabled from step {state.step + 1}."
                )

        # VapGPT warm-up: protect pretrained GPT/vap_head weights during early steps
        if (args.backchannel.enable
                and args.backchannel.module_type == "vap_gpt"
                and args.backchannel.bc_warmup_steps > 0
                and state.step <= args.backchannel.bc_warmup_steps):
            for name, p in model.named_parameters():
                if p.grad is not None and (
                    "backchannel.ar_channel" in name
                    or "backchannel.ar." in name
                    or "backchannel.vap_head" in name
                ):
                    p.grad.zero_()
            if state.step == args.backchannel.bc_warmup_steps and get_rank() == 0:
                logger.info(f"[step {state.step}] VapGPT warm-up done — GPT/vap_head gradients re-enabled.")

        if fusion_monitor is not None:
            fusion_monitor.before_update()
        optimizer.step()
        if fusion_monitor is not None:
            raw_aux = {"vap": float(vap_loss_val.item()), "vad": float(silence_loss_val.item()),
                       "bc": float(bc_event_loss_val.item())}
            coefficients = {"vap": args.backchannel.vap_loss_weight,
                            "vad": args.backchannel.vad_loss_weight,
                            "bc": args.backchannel.bc_event_loss_weight}
            auxiliary = {k: {"raw": raw_aux[k], "weighted": raw_aux[k] * coefficients[k],
                             "coefficient": coefficients[k], "valid_label_count": aux_coverage[k],
                             "target_mode": aux_mode[k]} for k in raw_aux}
            dynamics_stop = fusion_monitor.after_update(state.step, auxiliary)
            # Make the gate a normal evaluation/checkpoint boundary before exiting.
            is_last_step = is_last_step or dynamics_stop
        optimizer.zero_grad(set_to_none=True)  # 메모리 즉시 해제

        current_lrs = scheduler.get_last_lr()
        last_lr = current_lrs[0]
        scheduler.step()

        loss_item = loss.item()
        avg_loss = avg_aggregate(loss_item)
        if args.backchannel.enable:
            state.this_vap_loss = avg_aggregate(vap_loss_val.item())
            # state.this_commitment_loss = avg_aggregate(commitment_loss_val.item())
        if args.face_gen.enable:
            state.this_face_loss = avg_aggregate(face_loss_val.item())

        # 평가
        did_eval = args.do_eval and (
            (args.eval_freq > 0 and state.step % args.eval_freq == 0) or is_last_step
        )
        if did_eval:
            eval_data_loader = build_data_loader(
                instruct_tokenizer=interleaved_tokenizer,
                args=args.data,
                batch_size=args.batch_size,
                seed=args.seed if fusion_monitor is not None else None,
                rank=get_rank(),
                world_size=get_world_size(),
                is_eval=True,
            )
            evaluate(model, eval_data_loader, state, args,
                     mimi=mimi, face_codec=face_codec,
                     T_p=T_p, prompt_ids=prompt_ids if T_p else None,
                     param_dtype=param_dtype)
            del eval_data_loader

            eval_logs = get_eval_logs(
                state.step,
                avg_loss,
                state.this_eval_perplexity,
                state.this_eval_loss,
            )
            # [EPAD] eval metric (acc/precision/recall/f1)을 eval 로그에 포함
            if state.this_eval_epad_metrics is not None:
                eval_logs.update(state.this_eval_epad_metrics)
            main_logger_info(eval_log_msg(eval_logs))
            eval_logger.log(eval_logs, step=state.step)
            if fusion_monitor is not None:
                fusion_monitor.record_validation(state.step, state.this_text_loss,
                                                 getattr(state, "this_fusion_diagnostics", {}))

        # 타이밍
        state.end_step(n_batch_tokens)

        if args.backchannel.enable and bc_stats_accum is not None and state.step % args.log_freq == 0:
            state.this_bc_stats = {k: avg_aggregate(v.item()) for k, v in bc_stats_accum.items()}

        if state.step % args.log_freq == 0:
            if args.face_gen.enable and get_rank() == 0:
                total_mb = args.num_microbatches
                if face_loss_skipped_no_data > 0:
                    logger.warning(
                        f"[step {state.step}] face_loss skipped in "
                        f"{face_loss_skipped_no_data}/{total_mb} microbatch(es): "
                        f"batch.face_motion_gt was None — at least one sample had no "
                        f"matching FLAME .npy file (check flame_root path and file naming)."
                    )
                if face_loss_skipped_nonfinite > 0:
                    logger.warning(
                        f"[step {state.step}] face_loss skipped in "
                        f"{face_loss_skipped_nonfinite}/{total_mb} microbatch(es): "
                        f"face_loss was NaN or Inf."
                    )
            train_logs = get_train_logs(
                state,
                avg_loss,
                n_real_tokens,
                last_lr,
                torch.cuda.max_memory_allocated(),
                torch.cuda.memory_allocated(),
                args,
                vap_loss=state.this_vap_loss,   # → wandb "vap_loss" (None when backchannel disabled)
                face_loss=state.this_face_loss, # → wandb "face_loss" (None when face generation disabled)
            )
            # [EPAD] 예측 metric: rank별 confusion counts를 합산한 뒤 acc/recall/f1 계산
            epad_counts_global = epad_counts.clone()
            dist.all_reduce(epad_counts_global, op=dist.ReduceOp.SUM)
            train_logs.update(epad_metrics_from_counts(epad_counts_global, prefix="epad"))
            main_logger_info(train_log_msg(state, logs=train_logs, loss=avg_loss))
            if get_rank() == 0:
                logger.info(
                    f"[step {state.step}] [EPAD] train "
                    f"acc={train_logs['epad_acc']:.4f} "
                    f"precision={train_logs['epad_precision']:.4f} "
                    f"recall={train_logs['epad_recall']:.4f} "
                    f"f1={train_logs['epad_f1']:.4f}"
                )
            # VAP-module auxiliary losses → wandb. avg_aggregate is a collective
            # (all-reduce), so it must run on every rank — keep it outside any rank guard.
            if args.backchannel.enable:
                if args.backchannel.bc_event_loss_weight > 0:
                    train_logs["bc_event_loss"] = avg_aggregate(bc_event_loss_val.item())
                    # Running LA prior estimate (rank-local; converges across ranks).
                    _pri = (bc_class_counts / bc_class_counts.sum())
                    train_logs["bc_prior_epad"] = float(_pri[1])
                    train_logs["bc_prior_word"] = float(_pri[2])
                    # BC-head confusion against the active bc_target_mode target.
                    bc_head_global = bc_head_counts.clone()
                    dist.all_reduce(bc_head_global, op=dist.ReduceOp.SUM)
                    train_logs.update(
                        epad_metrics_from_counts(bc_head_global, prefix="bc_head")
                    )
                if args.backchannel.vad_loss_weight > 0:
                    train_logs["vad_loss"] = avg_aggregate(silence_loss_val.item())
                if args.backchannel.turn_end_floor_weight > 0:
                    train_logs["turn_end_floor_loss"] = avg_aggregate(turn_end_floor_val.item())
                    train_logs["turn_end_frame_frac"] = avg_aggregate(turn_end_frac_val.item())
            if args.face_gen.enable:
                # Raw face-loss components make imbalance/collapse visible in W&B.
                # avg_aggregate performs a distributed collective, so every rank
                # must execute this block before only rank 0 writes the metrics.
                for name, value in face_component_vals.items():
                    train_logs[f"face/{name}"] = avg_aggregate(value.item())
                train_logs["face/weighted_loss"] = (
                    0.0 if face_warmup_active else
                    args.face_gen.face_loss_weight * state.this_face_loss
                )
                train_logs["face/warmup_active"] = int(face_warmup_active)
                train_logs["lr/face_core"] = current_lrs[1]
                train_logs["lr/face_llm_proj"] = current_lrs[2]
            if teacher_model is not None:
                for name, value in kd_vals.items():
                    train_logs[f"continual/{name}_kd"] = avg_aggregate(
                        value.item()
                    )
            metrics_logger.log(train_logs, step=state.step)

        # 주기적 CUDA 캐시 비우기 (메모리 단편화 방지)
        if state.step % 200 == 0:
            torch.cuda.empty_cache()

        # 체크포인트 저장
        # best 모드(ckpt_keep_best_metric 설정)에서는 eval로 EPAD 점수가 갱신된
        # 스텝에만 저장하고 점수 상위 N개만 유지. 그 외엔 ckpt_freq/last_step 기준.
        epad_score = None
        if args.ckpt_keep_best_metric is not None:
            metric = args.ckpt_keep_best_metric
            # Resolve the selection score. prune_by_best keeps the top-N by score
            # (higher = better). EPAD metrics are higher-is-better → used directly.
            # Loss metrics are lower-is-better → negated so the LOWEST loss wins.
            # Only eligible on eval steps, when the metrics are freshly computed.
            sel_score = None
            if did_eval:
                if (state.this_eval_epad_metrics is not None
                        and metric in state.this_eval_epad_metrics):
                    sel_score = state.this_eval_epad_metrics[metric]
                elif metric == "text_loss" and state.this_text_loss is not None:
                    sel_score = -state.this_text_loss
                elif metric == "audio_loss" and state.this_audio_loss is not None:
                    sel_score = -state.this_audio_loss
                elif metric in ("eval_loss", "loss") and state.this_eval_loss is not None:
                    sel_score = -state.this_eval_loss
            save_now = args.do_ckpt and sel_score is not None
            if save_now:
                epad_score = sel_score
        else:
            save_now = args.do_ckpt and (
                (args.ckpt_freq > 0 and state.step % args.ckpt_freq == 0) or is_last_step
            )

        if save_now:
            torch.cuda.empty_cache()
            checkpointer.save_checkpoint(
                save_only_lora=args.lora.enable,
                dtype=param_dtype,
                score=epad_score,
            )
        if fusion_monitor is not None and (dynamics_stop or state.step == args.max_steps):
            fusion_monitor.finish(state.step, dynamics_stop)
            if dynamics_stop:
                main_logger_info("Fusion dynamics: static at step 500; checkpoint saved, advancing queue")
            break

    main_logger_info("학습 완료!")


if __name__ == "__main__":
    """사용법: torchrun --nproc_per_node=<N_GPUS> train.py config/example.yaml"""
    # torchrun --nproc_per_node=1 --master_port=29512 train.py config/example.yaml
    fire.Fire(train)
