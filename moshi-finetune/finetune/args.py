import logging
import os
from dataclasses import dataclass, field

from simple_parsing.helpers import Serializable

from .data.args import DataArgs


@dataclass
class BackchannelArgs(Serializable):
    """Backchannel / VAP 모듈 설정.

    module_type 선택:
      "mlp"     – 기존 BackchannelModule (LM 내부 MLP 기반 VAP + 토큰 게이팅).
      "vap_gpt" – 외부 VapGPT 모델을 보조 VAP 손실로 사용 (LM 내 BackchannelModule 비활성).
    """
    enable: bool = False
    module_type: str = "mlp"  # "mlp" | "vap_gpt"

    # ── MLP 모드 전용 ─────────────────────────────────────────────────────
    vap_dim: int = 256
    bc_hidden: int = 512
    gumbel_temp_init: float = 1.0
    gumbel_temp_min: float = 0.5
    gumbel_anneal_rate: float = 0.0001
    pad_token_id: int | None = None   # None이면 card (= initial_token_id) 사용
    epad_token_id: int | None = None  # None이면 card - 1 사용

    # ── 공통 ──────────────────────────────────────────────────────────────
    vap_loss_weight: float = 0.1
    commitment_loss_weight: float = 0.05  # Alt 3: bc_mlp + silence_gate alignment with depformer

    # Direct BC event supervision (focal BCE against EPAD ground truth)
    bc_event_loss_weight: float = 0.3      # weight for focal BCE on bc_mlp vs EPAD events
    bc_focal_gamma: float = 2.0            # focal loss exponent (down-weights easy negatives)
    bc_focal_pos_weight: float = 15.0      # pos_weight in BCE (~97/3 ratio, tuned down)

    # Direct silence-gate supervision (BCE against "user is silent" ground truth).
    # Target = user NOT speaking, derived from VAP label bit 7. Roughly balanced,
    # so plain BCE (no focal / pos_weight) is used. 0 disables. (v1 module only)
    silence_loss_weight: float = 0.3

    # v2 module (vap_gpt_module / lm2 / train2): current-frame VAD supervision.
    # BCE(vad_logits, per-frame energy-VAD targets from the stereo waveform),
    # both streams (user, agent). 0 disables.
    vad_loss_weight: float = 0.3

    # Learn START-vs-rest fusion jointly with the text model.  During training,
    # BC/VAP/VAD evidence is added as a differentiable residual to the EPAD
    # text logit; the normal text CE therefore calibrates the fusion weights.
    fusion_trainable: bool = True
    fusion_bc_init: float = 0.0
    fusion_vap_init: float = 0.0
    fusion_vad_init: float = 0.0
    fusion_bias_init: float = 0.0
    fusion_lr: float = 1e-3
    # Keep backchannel_fusion_* in fp32 (bf16 rounding silently drops small Adam
    # updates of these scalars). Single-GPU only.
    fusion_fp32: bool = False
    # Listener outputs that enter the EPAD residual: comma list of bc, vap, vad, decision.
    fusion_evidence: str = "bc,vap,vad"
    # Learned scale a on the backbone EPAD-vs-PAD log-odds: residual += (a - 1) * s.
    fusion_calibration: bool = False
    fusion_calibration_init: float = 1.0
    # Decision head: listener evidence trained only by the fused EPAD loss.
    decision_head: bool = False
    decision_hidden: int = 128
    decision_lr: float | None = None   # None → optim.lr
    # "full": VAP/VAD/BC losses also update the shared listener trunk (original).
    # "detach": they train only their own heads; the trunk follows the EPAD loss.
    aux_to_trunk: str = "full"
    # > 0: no EPAD residual while the agent is mid-utterance (a non-PAD text token in
    # the previous N frames); the same gate is applied at generation time.
    fusion_gate_frames: int = 0
    # Turn-end floor: hinge penalty weight * mean(relu(-residual)) on frames where the
    # user stopped speaking within the last turn_end_window frames and the agent is
    # silent, so the listener cannot learn to veto the backbone's own turn starts.
    turn_end_floor_weight: float = 0.0
    turn_end_window: int = 10

    # v2.1 BC target: boundary-ignore radius (frames). PAD frames within ±K of a
    # true [EPAD] token are set to ignore (-100) in the 3-class CE — onset labels
    # carry ±1–2 frame alignment jitter, and punishing near-misses teaches the
    # model to hedge (collapses p_epad toward the base rate). 0 disables.
    bc_onset_ignore_frames: int = 2

    # v2.1 BC loss: Logit-Adjustment temperature τ (Menon et al., ICLR 2021).
    # Training CE is computed on (logits + τ·log π) with running class priors π,
    # so plain argmax at inference approximates argmax P(c|x)/π_c^τ — no
    # inference-side correction needed. τ=0 → plain CE (under-fires the rare
    # EPAD class); τ=1 → balanced rule (over-fires). Sweep ~{0.5, 0.75, 1.0}
    # selecting by eval epad_f1. Replaces bc_focal_pos_weight in the v2.1 path
    # (that field remains for the v1 trainer only).
    bc_la_tau: float = 0.75

    # ── BC head supervision target (v2.1+) ────────────────────────────────
    # bc_logits [B, T, 3]가 무엇을 예측하도록 학습할지 선택합니다. 두 모드 모두
    # 같은 3-class head / Logit-Adjustment CE를 쓰므로 아키텍처 변경 없이 교체
    # 가능하고, 체크포인트 선택 지표인 epad_f1(텍스트 head 기준)은 모드와 무관
    # 하게 계산되므로 두 실험을 그대로 비교할 수 있습니다.
    #
    #   "epad" – GT 텍스트 토큰 기준 (기존 동작).
    #            0=PAD, 1=EPAD(실제 backchannel onset), 2=WORD(실제 단어).
    #            데이터셋 에이전트가 실제로 낸 반응을 그대로 모사합니다.
    #
    #   "vap"  – VAP manifest 기준 (turn-taking 동역학).
    #            2=WORD : 에이전트가 지금 발화 중 (현재 프레임 VAD, agent 채널)
    #            1=BC   : 지금은 침묵 + 향후 bc_vap_horizon_bins 구간 내 발화 시작
    #                     (+ bc_vap_require_user_silent면 같은 구간에 유저 침묵)
    #            0=PAD  : 그 외
    #            텍스트 라벨이 없는 데이터에도 적용되고, 실제 반응 여부와 무관하게
    #            "반응해도 되는 자리"를 학습합니다 (positive가 더 조밀함).
    bc_target_mode: str = "epad"  # "epad" | "vap"

    # "vap" 모드 전용: 미래 몇 개의 VAP bin까지를 "곧 발화"로 볼지 (1–4).
    # bin 경계는 누적 기준 0-200 / 200-600 / 600-1200 / 1200-2000ms 입니다.
    #   1 → 다음 200ms,  2 → 다음 600ms,  3 → 1.2s,  4 → 2.0s
    bc_vap_horizon_bins: int = 1
    # "vap" 모드 전용: 같은 horizon 구간에 유저가 침묵일 때만 BC(class 1)로 셀지.
    # True면 유저 발화와 겹치는 구간이 제외되어 positive가 희소해집니다.
    bc_vap_require_user_silent: bool = True

    # VapGPT warm-up: freeze GPT layers for this many steps so projections stabilise first
    bc_warmup_steps: int = 200

    # ── VapGPT 모드 전용 ──────────────────────────────────────────────────
    # VoiceActivityProjection 리포지토리 루트 경로
    vap_gpt_repo_path: str = "/home2/s20235100/Conversational-AI/VoiceActivityProjection"
    # 사전학습 VapGPT 체크포인트 (.pt 또는 .ckpt).
    # 기본값: VAP 리포에 포함된 공개 사전학습 가중치 (.pt = clean state dict).
    # None으로 설정하면 랜덤 초기화 + CPC 단독 사전학습 가중치만 사용.
    vap_gpt_checkpoint: str | None = (
        "/home2/s20235100/Conversational-AI/VoiceActivityProjection"
        "/example/VAP_3mmz3t0u_50Hz_ad20s_134-epoch9-val_2.56.pt"
    )
    # True면 VapGPT 파라미터 고정 (추론 전용; LM만 업데이트됨)
    vap_gpt_freeze: bool = True
    # VapGPT 내부 프레임률 (기본 50 fps)
    vap_gpt_frame_hz: int = 50
    # VapGPT 가 기대하는 오디오 샘플레이트 (Hz)
    vap_gpt_sample_rate: int = 16000
    # GPT 아키텍처 하이퍼파라미터 (사전학습 모델과 일치해야 함)
    vap_gpt_dim: int = 256
    vap_gpt_channel_layers: int = 1
    vap_gpt_cross_layers: int = 3
    vap_gpt_num_heads: int = 4
    vap_gpt_dropout: float = 0.1

    use_silence_ctx_proj: bool = True
    # CPC 인코더 고정 여부 (사전학습 가중치 사용 시 True 권장)
    # checkpoint_path가 설정된 경우 CPC는 체크포인트에서 로드되므로 별도 다운로드 없음.
    vap_gpt_freeze_encoder: bool = True
    # 대안 체크포인트: Lightning .ckpt 포맷도 지원
    # "/home2/s20235100/Conversational-AI/VoiceActivityProjection/example/50hz_48_10s-epoch20-val_1.85.ckpt"

    def __post_init__(self) -> None:
        if self.bc_target_mode not in ("epad", "vap"):
            raise ValueError(
                f"backchannel.bc_target_mode must be 'epad' or 'vap', "
                f"got {self.bc_target_mode!r}"
            )
        if not 1 <= self.bc_vap_horizon_bins <= 4:
            raise ValueError(
                f"backchannel.bc_vap_horizon_bins must be in [1, 4], "
                f"got {self.bc_vap_horizon_bins}"
            )
        ev = {e.strip() for e in self.fusion_evidence.split(",") if e.strip()}
        if ev - {"bc", "vap", "vad", "decision"}:
            raise ValueError(f"backchannel.fusion_evidence has unknown entries: {self.fusion_evidence!r}")
        if "decision" in ev and not self.decision_head:
            raise ValueError("backchannel.fusion_evidence 'decision' requires backchannel.decision_head=true")
        if self.aux_to_trunk not in ("full", "detach"):
            raise ValueError(f"backchannel.aux_to_trunk must be 'full' or 'detach', got {self.aux_to_trunk!r}")
        if self.fusion_gate_frames < 0 or self.turn_end_window < 1 or self.turn_end_floor_weight < 0:
            raise ValueError("backchannel.fusion_gate_frames >= 0, turn_end_window >= 1, turn_end_floor_weight >= 0")
        if self.turn_end_floor_weight > 0 and not self.fusion_trainable:
            raise ValueError("backchannel.turn_end_floor_weight needs backchannel.fusion_trainable=true")


@dataclass
class LoraArgs(Serializable):
    """Personaplex LMModel은 LoRA 레이어를 내장하지 않으므로,
    enable=False / full_finetuning=True 조합이 기본입니다.
    향후 LoRA 주입 지원 시 이 클래스를 확장할 수 있습니다."""
    enable: bool = False
    rank: int = 64
    scaling: float = 2.0
    ft_embed: bool = False  # True이면 임베딩 레이어도 학습

    def __post_init__(self) -> None:
        if self.enable:
            assert self.rank > 0
            assert self.scaling > 0.0


@dataclass
class OptimArgs(Serializable):
    lr: float = 1e-5
    weight_decay: float = 0.1
    pct_start: float = 0.05


@dataclass
class ContinualLearningArgs(Serializable):
    """Frozen PersonaPlex online-teacher distillation settings."""
    # New training default: retain the original PersonaPlex behaviour with an
    # online frozen teacher. Ablation launchers still write this field
    # explicitly so CL-off controls cannot change when defaults change again.
    enable: bool = True
    # None uses moshi_paths.moshi_path, i.e. the original PersonaPlex weights.
    # 로컬 파일 경로, 체크포인트 디렉터리, 또는 HF repo id("org/name") 모두 허용.
    teacher_checkpoint: str | None = None
    # HF repo id를 직접 지정할 때 사용 (예: "nvidia/personaplex-7b-v1").
    # 설정 시 teacher_checkpoint보다 우선하며 model.safetensors를 자동 다운로드.
    hf_repo_id: str | None = None
    temperature: float = 2.0
    text_kd_weight: float = 0.35
    speech_activity_kd_weight: float = 1.0
    turn_boundary_kd_weight: float = 0.25
    hidden_kd_weight: float = 0.05

    def __post_init__(self) -> None:
        if self.enable:
            assert self.temperature > 0.0
            for name in (
                "text_kd_weight",
                "speech_activity_kd_weight",
                "turn_boundary_kd_weight",
                "hidden_kd_weight",
            ):
                assert getattr(self, name) >= 0.0


@dataclass
class WandbArgs(Serializable):
    project: str | None = None  # None이면 wandb 미사용
    offline: bool = False
    key: str | None = None
    run_name: str | None = None
    local_dir: str = "/dev/shm/wandb"  # wandb staging 파일 저장 위치 (/tmp은 용량 부족, RAM tmpfs 사용)

    def __post_init__(self) -> None:
        if self.project is not None:
            try:
                import wandb  # noqa: F401
            except ImportError:
                raise ImportError(
                    "`wandb` not installed. Either make sure `wandb` is installed or set `wandb:project` to None."
                )
            if len(self.project) == 0:
                raise ValueError("`wandb.project` must not be an empty string.")


@dataclass
class ModelPaths(Serializable):
    """Personaplex 모델 가중치 경로 설정.
    local path를 지정하거나, hf_repo_id만 지정해 HuggingFace에서 자동 다운로드합니다.
    """
    hf_repo_id: str | None = "nvidia/personaplex-7b-v1"
    mimi_path: str | None = None      # tokenizer-e351c8d8-checkpoint125.safetensors
    moshi_path: str | None = None     # model.safetensors
    tokenizer_path: str | None = None # tokenizer_spm_32k_3.model
    config_path: str | None = None    # 사용자 정의 LM 설정 JSON (선택)


@dataclass
class FaceGenArgs(Serializable):
    """Face generation module settings (CausalSoftVQContinuousTransformer).

    Controlled by face_gen.enable in config.  When disabled the face module
    is never loaded, so no checkpoint path is required.
    """
    enable: bool = False

    # Path to a CausalSoftVQContinuousTransformer checkpoint (.ckpt).
    # Must be set when enable=True.
    ckpt_path: str | None = None

    # Face model version: 1 = softvq_continuous_online_train (original),
    # 2 = softvq_continuous_online_train_v2 (blink + block-causal chunks + MTP
    # look-ahead). Must match the checkpoint's training script — v1/v2 state
    # dicts are not interchangeable.
    model_version: int = 1

    # Model architecture — must match the saved checkpoint.
    # Defaults are read from the checkpoint's saved args when ckpt_path is given,
    # so these only need to be set if the checkpoint lacks an "args" key.
    hidden_dim: int = 512
    layers: int = 6
    heads: int = 8
    code_dim: int = 32
    prior_warmup_frames: int = 10

    # Sliding context window (in face frames at 25 fps) for autoregressive generation.
    max_context_frames: int = 25

    # When True, the audio feature fed to the face module is decoded from the
    # model's own predicted audio codes (argmax of depformer logits) rather than
    # from the ground-truth codes (teacher forcing).
    use_generated_audio_feat: bool = False
    # Keep the v7.2 face model trainable, but prevent face reconstruction loss
    # from changing the conversational backbone through its LLM feature input.
    detach_llm_features: bool = False

    # ── ARTalkCodec (VAE) for z-space loss computation ────────────────────
    # Frozen codec used only to compute z_target = quant_to_sum_feat(gt_face_motion).
    codec_ckpt_path: str | None = None   # e.g. /home6/duplex/VAE_ami_dualtalk/checkpoints/iter_100000.pt
    codec_stats_path: str | None = None  # e.g. /home6/duplex/dataset/artalk_mimi54_stats.json
    # Codec architecture — must match the saved codec checkpoint.
    codec_hidden_dim: int = 512
    codec_depth: int = 8
    codec_heads: int = 8

    # ── FLAME / 3DMM data root ─────────────────────────────────────────────
    # Root directory containing FLAME .npy files structured as:
    #   {flame_root}/{speaker}/{split}/{stem}_{speaker}.npy
    # where speaker ∈ {"bc", "ut"} and split ∈ {"train", "valid", "test"}.
    flame_root: str = ""
    # "auto": AMI original → bc, AMI *_switch → ut.
    flame_speaker: str = "bc"  # "bc" | "ut" | "auto"

    # ── Per-component loss weights (from reference pretraining) ───────────
    # Overall weight applied to the sum of all face sub-losses.
    face_loss_weight: float = 1.0
    # Do not add face loss to the training objective for the first N steps.
    # Forward/loss computation and monitoring still run during this period.
    warmup_steps: int = 0
    # Separate rates for the pretrained face core and the newly connected
    # zero-initialized LLM projection. Set either to 0 to inherit optim.lr.
    core_lr: float = 0.0
    llm_proj_lr: float = 0.0
    # Exposure-bias mitigation: replace this fraction of GT previous-motion
    # frames with a no-grad self prediction, ramped in after warm-up.
    scheduled_sampling_prob: float = 0.0
    scheduled_sampling_ramp_steps: int = 0
    scheduled_sampling_keep_head_frames: int = 0
    # Sub-loss weights (matching softvq_continuous_online_train.py defaults).
    motion_weight: float = 1.0       # L1 on pred_motion vs gt
    prior_weight: float = 0.5        # L1 on prior_motion vs gt
    z_weight: float = 0.2            # MSE on z_pred vs z_target (from codec)
    z_bce_weight: float = 0.1        # binary-CE on z_pred bits vs z_target bits
    jaw_weight: float = 2.0          # extra L1 weight on jaw dimension [50:51]
    vel_weight: float = 0.5          # MSE on frame velocity
    reg_weight: float = 0.02         # regularisation on delta and residual magnitude
    gate_weight: float = 1.0         # gate target MSE loss
    gate_target_expr: float = 0.45
    gate_target_jaw: float = 0.25
    gate_target_neck: float = 0.65
    gate_loss_expr_weight: float = 4.0
    gate_loss_jaw_weight: float = 4.0
    gate_loss_neck_weight: float = 1.0


@dataclass
class FusionDynamicsArgs(Serializable):
    """Opt-in diagnostics; existing training runs keep their original behavior."""
    enable: bool = False
    gate_step: int = 500
    static_excursion: float = 0.02
    static_late_range: float = 0.01
    dynamic_excursion: float = 0.05
    dynamic_end_shift: float = 0.02
    stable_tail: float = 0.02


@dataclass
class TrainArgs(Serializable):
    data: DataArgs

    run_dir: str  # 체크포인트와 로그가 저장될 디렉토리 (존재하지 않아야 함)
    moshi_paths: ModelPaths = field(default_factory=ModelPaths)
    # Optional stage-2 model initialization. Loads model weights only; optimizer,
    # scheduler, and TrainState start fresh. Face-module tensors are deliberately
    # excluded so face_gen.ckpt_path remains the face initialization source.
    init_checkpoint_dir: str | None = None

    # 손실 가중치
    first_codebook_weight_multiplier: float = 1.0
    text_padding_weight: float = 0.5
    epad_weight: float = 1.0  # EPAD(end-of-text-padding) 토큰 손실 가중치 (PAD와 별도)

    # 텍스트 프롬프트 (에이전트 지시문)
    # 비어 있으면 프롬프트 없이 학습 (기존 동작 유지).
    # 설정 시 매 학습 스텝에서 codes의 앞에 접두사로 삽입되어
    # inference의 step_system_prompts() 와 동일한 방식으로 LM에 제공됩니다.
    text_prompt: str | list[str] | None = None

    optim: OptimArgs = field(default_factory=OptimArgs)
    seed: int = 0
    num_microbatches: int = 1   # 그래디언트 누적 횟수

    duration_sec: float = 10.0  # 학습 세그먼트 길이(초)
    batch_size: int = 1
    max_norm: float = 1.0       # 그래디언트 클리핑 최대 norm
    max_steps: int = 1000
    log_freq: int = 10

    # 체크포인트
    ckpt_freq: int = 0          # 0이면 마지막 스텝에만 저장
    save_adapters: bool = True
    do_ckpt: bool = True
    num_ckpt_keep: int | None = 3

    # Best-checkpoint 유지 (EPAD 예측 점수 기준)
    # None이면 기존 recency(최근 num_ckpt_keep개) 정책을 사용.
    # 설정 시(예: "epad_f1") eval에서 해당 metric이 갱신된 스텝에만 저장하고
    # 점수 상위 ckpt_keep_best_n개만 남깁니다.
    # 지원 metric:
    #   - higher-is-better: epad_f1 / epad_recall / epad_precision / epad_acc
    #   - lower-is-better (내부적으로 부호 반전해 최저값 유지): text_loss / audio_loss / eval_loss
    ckpt_keep_best_metric: str | None = None
    ckpt_keep_best_n: int = 3

    # 평가
    eval_freq: int = 0
    do_eval: bool = False

    # 효율
    gradient_checkpointing: bool = True

    world_size: int | None = field(init=False, default=None)

    # 로깅
    wandb: WandbArgs = field(default_factory=WandbArgs)

    # LoRA (Personaplex는 기본 비활성화)
    lora: LoraArgs = field(default_factory=LoraArgs)
    # Personaplex는 LoRA 미지원이므로 full_finetuning=True가 기본값
    full_finetuning: bool = True
    # VAP-only ablation: freeze the Personaplex backbone and train only the
    # backchannel module (VAP GPT + BC/VAD heads).
    freeze_backbone: bool = False
    freeze_depformer: bool = False  # True이면 Depformer 전체를 동결 (LM transformer만 학습)

    # Backchannel VAP
    backchannel: BackchannelArgs = field(default_factory=BackchannelArgs)
    fusion_dynamics: FusionDynamicsArgs = field(default_factory=FusionDynamicsArgs)

    # Face generation (inference-only; not used during training)
    face_gen: FaceGenArgs = field(default_factory=FaceGenArgs)
    continual_learning: ContinualLearningArgs = field(
        default_factory=ContinualLearningArgs
    )

    param_dtype: str = "bfloat16"
    overwrite_run_dir: bool = False

    def __post_init__(self) -> None:
        assert getattr(self, "world_size", None) is None
        self.world_size = int(os.environ.get("WORLD_SIZE", -1))

        if self.wandb.offline:
            command = f"cd {self.run_dir}; wandb sync --sync-all"
            logging.info(f"to sync wandb offline, run: {command}")

        assert self.num_microbatches >= 1
        assert self.num_ckpt_keep is None or self.num_ckpt_keep >= 1
        if self.fusion_dynamics.enable:
            if self.world_size != 1:
                raise ValueError("Fusion dynamics monitoring currently requires WORLD_SIZE=1")
            if not self.backchannel.enable or not self.backchannel.fusion_trainable:
                raise ValueError("Fusion dynamics requires enabled, trainable backchannel fusion")
            if self.max_steps != 1000 or self.fusion_dynamics.gate_step != 500:
                raise ValueError("Fusion dynamics uses the reviewed 1000-step horizon and 500-step gate")
            if not self.do_ckpt or self.ckpt_keep_best_metric is not None:
                raise ValueError("Fusion dynamics requires periodic checkpoints, not best-only pruning")
            if self.ckpt_freq != 100 or self.num_ckpt_keep is None or self.num_ckpt_keep < 10:
                raise ValueError("Fusion dynamics requires 100-step checkpoints and retention >=10")


        # Personaplex: LoRA 및 full_finetuning 지원
        if (
            not self.lora.enable
            and not self.full_finetuning
            and not self.freeze_backbone
        ):
            logging.warning(
                "LoRA is disabled and full_finetuning is False. "
                "Forcing full_finetuning=True for Personaplex model."
            )
            self.full_finetuning = True
