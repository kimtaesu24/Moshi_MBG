import functools
import logging
import math
from typing import Callable, Union

import safetensors
import torch
import torch.distributed.fsdp.wrap as torch_wrap

# personaplex_MBG/moshi 패키지에서 임포트
from moshi.models.lm import LMModel, ScaledEmbedding
from moshi.models.loaders import get_moshi_lm, _lm_kwargs
from moshi.models.backchannel_vap import BackchannelModule
from moshi.models.vap_gpt_module import VapGPTBackchannelModule
from moshi.modules.transformer import StreamingTransformerLayer, RMSNorm

from torch.distributed.fsdp import BackwardPrefetch
from torch.distributed.fsdp.api import ShardingStrategy
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel

from .args import TrainArgs
from .distributed import get_rank, get_world_size

logger = logging.getLogger(__name__)


def main_logger_info(message: str) -> None:
    if get_rank() == 0:
        logger.info(message)


def get_fsdp_policy(freeze_non_embed: bool) -> Callable[[torch.nn.Module], bool]:
    """FSDP wrap 정책 반환.

    - StreamingTransformerLayer: 자체 FSDP 그룹
    - ScaledEmbedding, RMSNorm: 자체 FSDP 그룹
    """

    def _should_wrap(module: torch.nn.Module) -> bool:
        if isinstance(module, (
            StreamingTransformerLayer,
            ScaledEmbedding,
            RMSNorm,
            BackchannelModule,
            VapGPTBackchannelModule,
        )):
            return True
        # Face module: CausalSoftVQContinuousTransformer is dynamically imported at
        # runtime (its path is set in lm.py __init__), so we check by class name.
        if type(module).__name__ == "CausalSoftVQContinuousTransformer":
            return True
        return False

    base_policy = functools.partial(
        torch_wrap.lambda_auto_wrap_policy, lambda_fn=_should_wrap
    )

    if not freeze_non_embed:
        return base_policy

    def fsdp_mixed_grad_policy_fn(module):
        return all(p.requires_grad for p in module.parameters())

    fsdp_mixed_policy = functools.partial(
        torch_wrap.lambda_auto_wrap_policy, lambda_fn=fsdp_mixed_grad_policy_fn
    )

    policies = [fsdp_mixed_policy, base_policy]
    return functools.partial(torch_wrap._or_policy, policies=policies)


def log_train_params(model: Union[torch.nn.Module, FullyShardedDataParallel]):
    world_size = get_world_size()
    num_params = world_size * sum(p.numel() for p in model.parameters())
    num_train_params = world_size * sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    main_logger_info(
        f"{num_train_params:,.0f} / {num_params:,.0f} 파라미터 학습 중 "
        f"({num_train_params / num_params * 100:.2f}%)"
    )


def get_fsdp_model(
    args: TrainArgs,
    moshi_weight: str,
    lm_kwargs: dict,
) -> Union[FullyShardedDataParallel, LMModel]:
    """Personaplex LMModel을 초기화하고 FSDP로 샤딩하여 반환합니다."""
    if args.param_dtype == "bfloat16":
        param_dtype = torch.bfloat16
    elif args.param_dtype == "float32":
        param_dtype = torch.float32
    else:
        raise ValueError(f"지원하지 않는 dtype: {args.param_dtype}")

    # get_moshi_lm 대신 LMModel을 직접 인스턴스화 (get_moshi_lm은 filename 문자열을 요구함)
    with torch.device("cpu"):
        model = LMModel(device="cpu", dtype=param_dtype, **lm_kwargs)

    main_logger_info("모델 구조 생성 완료, 가중치 로드 중...")

    # CPU RAM OOM 방지를 위해 Rank 0에서만 실제 가중치 로드
    # FSDP(sync_module_states=True)가 자동으로 설정된 파라미터를 다른 Rank로 브로드캐스트합니다.
    if get_rank() == 0:
        with safetensors.safe_open(moshi_weight, framework="pt", device="cpu") as f:
            state_dict = {k: f.get_tensor(k) for k in f.keys()}

        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        missing_base = [k for k in missing_keys if not k.startswith("backchannel")]
        if missing_base or unexpected_keys:
            raise RuntimeError(
                f"Moshi checkpoint mismatch: missing={missing_base}, unexpected={unexpected_keys}"
            )
        main_logger_info(f"Moshi backbone loaded exactly; {len(missing_keys)} ours parameters initialized")

        del state_dict # RAM 즉시 반환

    model.to(param_dtype)

    main_logger_info("가중치 로드 및 캐스팅 완료")

    torch.distributed.barrier()

    freeze_non_embed = False

    if args.freeze_backbone:
        if not args.backchannel.enable:
            raise ValueError(
                "freeze_backbone=True requires backchannel.enable=True; "
                "otherwise there are no VAP parameters to train."
            )
        for param in model.parameters():
            param.requires_grad = False
        for name, param in model.named_parameters():
            if "backchannel" in name:
                param.requires_grad = True
        trainable_backchannel = sum(
            p.numel() for name, p in model.named_parameters()
            if "backchannel" in name and p.requires_grad
        )
        main_logger_info(
            "Backbone frozen: training only backchannel/VAP parameters "
            f"({trainable_backchannel:,} parameters)"
        )

    elif args.lora.enable:
        from peft import get_peft_model, LoraConfig, TaskType
        
        main_logger_info(f"LoRA 모드 활성화 (Rank={args.lora.rank}, Scaling={args.lora.scaling})")
        # LLM transformer 레이어만 LoRA 타겟팅.
        # face_module / backchannel 은 full fine-tuning 이므로 제외.
        # PEFT는 target_modules가 문자열이면 re.fullmatch(pattern, full_module_key)로 매칭하므로
        # 네거티브 룩어헤드로 face_module/backchannel 경로를 명시적으로 배제한다.
        target_modules = (
            r"(?!.*(face_module|backchannel))"
            r".*(in_proj|out_proj|linear1|linear2|text_linear|input_proj|linear_in|linear_out)"
        )
        
        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=args.lora.rank,
            lora_alpha=args.lora.scaling * args.lora.rank,
            target_modules=target_modules,
            bias="none",
        )

        class PeftCompatibleWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def __getattr__(self, name):
                try:
                    return super().__getattr__(name)
                except AttributeError:
                    return getattr(self.model, name)

            def forward(self, input_ids=None, *args, **kwargs):
                # PEFT → input_ids로 넘김; PEFT-injected kwargs (attention_mask 등) 필터링
                # 실제 모델 → codes로 받음
                return self.model(input_ids, step=kwargs.get("step", 0),
                                  voice_prompt_embs=kwargs.get("voice_prompt_embs", None),
                                  audio_feat=kwargs.get("audio_feat", None),
                                  gt_face_motion=kwargs.get("gt_face_motion", None),
                                  mimi=kwargs.get("mimi", None),
                                  bc_audio_feats=kwargs.get("bc_audio_feats", None))

        model = PeftCompatibleWrapper(model)
        model = get_peft_model(model, peft_config)

        # PEFT injects lora_A, lora_B, and other adapter tensors in fp32 by default,
        # after the base model was already cast to param_dtype.  A single .to() here
        # ensures every parameter and buffer (including all PEFT additions) is in the
        # correct dtype before FSDP or cuda() is called.
        model = model.to(param_dtype)

        # Backchannel and face modules are trained fully (not via LoRA).
        # target_modules above won't match their layer names, but PEFT may
        # still freeze them — explicitly re-enable gradients here.
        for name, param in model.named_parameters():
            if "backchannel" in name or "face_module" in name:
                param.requires_grad = True

        if args.gradient_checkpointing and hasattr(model, "enable_input_require_grads"):
            # Gradient Checkpointing과 LoRA(인풋 requires_grad=False) 조합을 위한 설정
            model.enable_input_require_grads()
            
        if get_rank() == 0:
            model.print_trainable_parameters()
            
    elif args.full_finetuning:
        for param in model.parameters():
            param.requires_grad = True
        main_logger_info("Full finetuning: 모든 파라미터 학습 가능")
    elif args.lora.ft_embed:
        freeze_non_embed = True
        for name, param in model.named_parameters():
            if "emb" in name or "text_emb" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        main_logger_info("ft_embed 모드: 임베딩 레이어만 학습")
    else:
        for param in model.parameters():
            param.requires_grad = True
        main_logger_info("Full finetuning (fallback): 모든 파라미터 학습 가능")

    if args.freeze_depformer:
        frozen = sum(
            p.numel() for name, p in model.named_parameters() if "depformer" in name
        )
        for name, param in model.named_parameters():
            if "depformer" in name:
                param.requires_grad = False
        main_logger_info(f"Depformer 동결: {frozen:,}개 파라미터")

    if args.backchannel.enable and args.backchannel.fusion_fp32:
        # The fusion scalars are tiny and updated with a large LR; in bf16 most Adam
        # steps round to zero. Keep them (and only them) in fp32.
        if get_world_size() != 1:
            raise ValueError("backchannel.fusion_fp32 is supported on a single GPU only "
                             "(FSDP flat parameters need one dtype)")
        for name, param in model.named_parameters():
            if "backchannel_fusion_" in name:
                param.data = param.data.float()
        main_logger_info("Fusion parameters kept in fp32")

    if get_world_size() == 1:
        model = model.cuda()

        if args.gradient_checkpointing:
            from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
                checkpoint_wrapper,
                CheckpointImpl,
                apply_activation_checkpointing,
            )
            apply_activation_checkpointing(
                model,
                checkpoint_wrapper_fn=functools.partial(
                    checkpoint_wrapper,
                    checkpoint_impl=CheckpointImpl.NO_REENTRANT,
                ),
                check_fn=lambda m: isinstance(m, StreamingTransformerLayer),
            )
            main_logger_info("Gradient checkpointing 적용 완료 (단일 GPU)")

        log_train_params(model)
        return model

    auto_wrap_policy = get_fsdp_policy(freeze_non_embed)
    main_logger_info(f"{get_world_size()}개 GPU에 모델 샤딩 중...")

    wrapped_model = FullyShardedDataParallel(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        auto_wrap_policy=auto_wrap_policy,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        limit_all_gathers=True,
        device_id=torch.cuda.current_device(),
        sync_module_states=True,
        use_orig_params=True,
    )

    main_logger_info("모델 샤딩 완료!")

    if args.gradient_checkpointing:
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            checkpoint_wrapper,
            CheckpointImpl,
            apply_activation_checkpointing,
        )
        apply_activation_checkpointing(
            wrapped_model,
            checkpoint_wrapper_fn=functools.partial(
                checkpoint_wrapper,
                checkpoint_impl=CheckpointImpl.NO_REENTRANT,
            ),
            check_fn=lambda m: isinstance(m, StreamingTransformerLayer),
        )
        main_logger_info("Gradient checkpointing 적용 완료 (FSDP 후)")

    log_train_params(wrapped_model)
    return wrapped_model
