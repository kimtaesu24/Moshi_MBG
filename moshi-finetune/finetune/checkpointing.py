import json
import logging
import shutil
from pathlib import Path

import safetensors.torch
import torch
from moshi.models.lm import LMModel
try:
    from moshi.modules.lora import LoRALinear
    _has_lora = True
except ImportError:
    LoRALinear = None
    _has_lora = False
from torch.distributed import barrier
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel

from .distributed import get_rank, get_world_size
from .utils import TrainState

logger = logging.getLogger("checkpointing")


def main_logger_info(message: str) -> None:
    if get_rank() == 0:
        logger.info(message)


class Checkpointer:
    """A class to save PyTorch model and optimizer states"""

    def __init__(
        self,
        model: FullyShardedDataParallel | LMModel,
        state: TrainState,
        run_dir: Path | str,
        config: dict,
        optimizer: torch.optim.Optimizer | None = None,
        num_ckpt_keep: int | None = None,
        full_finetuning: bool = False,
        keep_best_metric: str | None = None,
        keep_best_n: int = 3,
        save_backchannel_only: bool = False,
    ):
        self.model = model
        self.optimizer = optimizer
        self.state = state
        self.run_dir = Path(run_dir)
        self.rank = get_rank()
        self.num_ckpt_keep = num_ckpt_keep
        self.full_finetuning = full_finetuning
        self.config = config
        # Best-checkpoint 유지: keep_best_metric 설정 시 점수 상위 keep_best_n개만 보존.
        self.keep_best_metric = keep_best_metric
        self.keep_best_n = keep_best_n
        self.save_backchannel_only = save_backchannel_only

    @property
    def ckpt_dir(self) -> Path:
        return self.run_dir / "checkpoints"

    @property
    def dst_dir(self) -> Path:
        return self.ckpt_dir / f"checkpoint_{self.state.step:06d}" / "consolidated"

    @staticmethod
    def consolidated_path(ckpt_dir: Path, save_only_lora: bool = False) -> Path:
        suffix = "safetensors"
        prefix = "lora" if save_only_lora else "consolidated"

        return ckpt_dir / f"{prefix}.{suffix}"

    @staticmethod
    def _tmp(ckpt_dir: Path) -> Path:
        return ckpt_dir.with_name(f"tmp.{ckpt_dir.name}")

    def delete_old_ckpts(self) -> list[Path]:
        all_saved_ckpts = [d for d in self.ckpt_dir.iterdir() if d.is_dir()]

        # Sort directories by creation time (oldest to newest)
        all_saved_ckpts.sort(key=lambda x: x.stat().st_ctime, reverse=True)

        ckpts_to_delete = all_saved_ckpts[self.num_ckpt_keep :]

        for ckpt_to_delete in ckpts_to_delete:
            try:
                shutil.rmtree(ckpt_to_delete)
                main_logger_info(f"Deleted ckpt: {ckpt_to_delete}")
            except OSError as e:
                main_logger_info(f"Error deleting directory {ckpt_to_delete}: {e}")

        return ckpts_to_delete

    @property
    def best_registry_path(self) -> Path:
        """JSON registry of {step: score} for best-checkpoint retention."""
        return self.ckpt_dir / "best_checkpoints.json"

    def prune_by_best(self, step: int, score: float) -> list[Path]:
        """Record *score* for *step* and keep only the top ``keep_best_n`` ckpts.

        Higher score = better. Checkpoints whose score drops out of the top-N
        (including the one just saved, if it doesn't qualify) are deleted along
        with their registry entry. Only meaningful on rank 0.
        """
        registry: dict[str, float] = {}
        if self.best_registry_path.exists():
            try:
                registry = json.loads(self.best_registry_path.read_text())
            except (json.JSONDecodeError, OSError):
                registry = {}
        registry[str(step)] = float(score)

        # Drop registry entries whose checkpoint dir no longer exists (manual rm).
        registry = {
            s: v
            for s, v in registry.items()
            if (self.ckpt_dir / f"checkpoint_{int(s):06d}").exists()
        }

        # Sort by score (desc); ties broken by newer step first.
        ordered = sorted(registry.items(), key=lambda kv: (kv[1], int(kv[0])), reverse=True)
        keep = dict(ordered[: self.keep_best_n])
        drop = ordered[self.keep_best_n :]

        deleted: list[Path] = []
        for step_str, _ in drop:
            ckpt_path = self.ckpt_dir / f"checkpoint_{int(step_str):06d}"
            if ckpt_path.exists():
                try:
                    shutil.rmtree(ckpt_path)
                    deleted.append(ckpt_path)
                except OSError as e:
                    main_logger_info(f"Error deleting directory {ckpt_path}: {e}")
            registry.pop(step_str, None)

        self.best_registry_path.write_text(json.dumps(registry, indent=2))
        kept_desc = ", ".join(
            f"step {s}={registry[s]:.4f}"
            for s in sorted(keep, key=lambda s: registry[s], reverse=True)
        )
        main_logger_info(
            f"[Checkpointer] best-{self.keep_best_n} by '{self.keep_best_metric}': {kept_desc}"
        )
        return deleted

    def write_params_info(self, tmp_dst: Path):
        params_path = tmp_dst / "config.json"
        with open(params_path, "w") as f:
            f.write(json.dumps(self.config, indent=4))

    @staticmethod
    def get_non_lora_states(
        state_dict: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        return {
            k: v
            for k, v in state_dict.items()
            if not any(l_key in k for l_key in ["lora", "frozen"])
        }

    # ──────────────────────────────────────────────────────────────────────
    # Helper: keys to save in hybrid LoRA + full-finetune mode.
    #
    # When the backbone is LoRA-finetuned but face_module / backchannel are
    # full-finetuned, we want to persist:
    #   1. All LoRA adapter tensors  (lora_A, lora_B, …)
    #   2. All face_module parameters
    #   3. All backchannel parameters
    #
    # We derive the set from the state_dict keys directly (avoiding the
    # FSDP flat-param name mismatch that occurs with named_parameters()).
    # ──────────────────────────────────────────────────────────────────────
    _FULL_FINETUNE_PREFIXES: tuple[str, ...] = ("face_module", "backchannel")
    _LORA_KEYWORDS: tuple[str, ...] = ("lora_A", "lora_B", "lora_embedding",
                                        "lora_magnitude", "lora_dropout")

    @classmethod
    def _is_hybrid_save_key(cls, key: str) -> bool:
        """Return True if *key* should be included in a hybrid LoRA checkpoint.

        Included:
          - LoRA adapter weights (lora_A / lora_B / …)
          - face_module.* and backchannel.* parameters (full fine-tune)
        Excluded:
          - frozen base-model weights (neither LoRA nor full-FT modules)
        """
        # LoRA adapter tensors
        if any(kw in key for kw in cls._LORA_KEYWORDS):
            return True
        # Full-finetune sub-modules
        # Normalise PEFT / FSDP path prefixes (e.g. "base_model.model.model.")
        # by checking *any* segment of the dot-split key.
        segments = key.split(".")
        for prefix in cls._FULL_FINETUNE_PREFIXES:
            if any(seg == prefix or seg.startswith(prefix) for seg in segments):
                return True
        return False

    @torch.no_grad()
    def retrieve_save_states(
        self, save_only_lora: bool, save_dtype: torch.dtype
    ) -> dict[str, torch.Tensor]:
        assert not (save_only_lora and self.full_finetuning), (
            "Cannot save LoRA checkpoint as LoRA training is not enabled."
        )

        # remove all potential hooks
        if _has_lora and LoRALinear is not None:
            for module in self.model.modules():
                if isinstance(module, LoRALinear) and hasattr(module, "_merge_lora_handle"):
                    module._merge_lora_handle.remove()  # type: ignore

        offload_to_cpu = get_world_size() > 1
        if self.save_backchannel_only:
            assert not save_only_lora, (
                "Backchannel-only saving and LoRA-only saving are mutually exclusive."
            )
            assert (
                isinstance(self.model, FullyShardedDataParallel)
                or get_world_size() == 1
            )

            def keep_dtype(key: str, value: torch.Tensor) -> torch.dtype:
                # fp32 fusion scalars (backchannel.fusion_fp32) are stored as fp32.
                if "backchannel_fusion_" in key and value.dtype == torch.float32:
                    return torch.float32
                return save_dtype

            def is_backchannel_key(key: str) -> bool:
                segments = key.split(".")
                return any(
                    segment == "backchannel"
                    or segment.startswith("backchannel_fusion_")
                    for segment in segments
                )

            if get_world_size() > 1:
                with self.model.summon_full_params(
                    self.model, writeback=True, offload_to_cpu=offload_to_cpu
                ):
                    full_sd = self.model.state_dict()
                    states = {
                        key: value.to(dtype=keep_dtype(key, value))
                        for key, value in full_sd.items()
                        if is_backchannel_key(key)
                    }
            else:
                full_sd = self.model.state_dict()
                states = {
                    key: value.to(device="cpu", dtype=keep_dtype(key, value))
                    for key, value in full_sd.items()
                    if is_backchannel_key(key)
                }
            if not states:
                raise RuntimeError("Backchannel-only checkpoint selected no tensors")
            main_logger_info(
                f"[Checkpointer] Backchannel-only save: {len(states)} tensors"
            )
        elif save_only_lora:
            # ── Hybrid mode: LoRA backbone + full-finetune face/VAP ──────────
            # Collect trainable state-dict keys by inspecting the state_dict
            # directly (avoids FSDP flat-param name mismatch).
            assert (
                isinstance(self.model, FullyShardedDataParallel)
                or get_world_size() == 1
            )
            if get_world_size() > 1:
                with self.model.summon_full_params(
                    self.model, writeback=True, offload_to_cpu=offload_to_cpu
                ):
                    full_sd = self.model.state_dict()
                    states = {
                        k: v.to(dtype=save_dtype)
                        for k, v in full_sd.items()
                        if self._is_hybrid_save_key(k)
                    }
            else:
                full_sd = self.model.state_dict()
                states = {
                    k: v.to(device="cpu", dtype=save_dtype)
                    for k, v in full_sd.items()
                    if self._is_hybrid_save_key(k)
                }

            main_logger_info(
                f"[Checkpointer] Hybrid save: {len(states)} tensors "
                f"(LoRA adapters + face_module + backchannel)"
            )
        else:
            # merge weights if we don't just save LoRA
            if _has_lora and LoRALinear is not None:
                def merge_lora(
                    m: torch.nn.Module,
                    destination: dict[str, torch.Tensor],
                    prefix: str,
                    *args,
                ):
                    weight = m.merge_weight()  # type: ignore
                    destination[prefix + "weight"] = weight

                for module in self.model.modules():
                    if isinstance(module, LoRALinear):
                        module._merge_lora_handle = module._register_state_dict_hook(
                            merge_lora
                        )

            # make sure you have enough CPU RAM available to save the full model
            assert (
                isinstance(self.model, FullyShardedDataParallel)
                or get_world_size() == 1
            ), (
                "`self.model` should be an instance of `FullyShardedDataParallel` if `world_size > 1`"
            )
            if get_world_size() > 1:
                with self.model.summon_full_params(
                    self.model, writeback=True, offload_to_cpu=offload_to_cpu
                ):
                    states = self.get_non_lora_states(self.model.state_dict())
                    states = {k: v.to(dtype=save_dtype) for k, v in states.items()}
            else:
                states = self.get_non_lora_states(self.model.state_dict())
                states = {k: v.to(device="cpu", dtype=save_dtype) for k, v in states.items()}

        states = dict(sorted(states.items()))
        return states

    @torch.no_grad()
    def save_checkpoint(
        self,
        save_only_lora: bool,
        dtype: torch.dtype = torch.float16,
        score: float | None = None,
    ):
        if self.full_finetuning:
            assert not save_only_lora, "Cannot save LoRA checkpoint in full finetuning"

        tmp_dst = self._tmp(self.dst_dir)
        main_logger_info(
            f"Dumping checkpoint in {self.dst_dir} using tmp name: {tmp_dst.name}"
        )

        assert not self.dst_dir.exists(), f"dst exists {self.dst_dir}"
        tmp_dst.mkdir(parents=True, exist_ok=True)

        with torch.no_grad():
            states: dict[str, torch.Tensor] = self.retrieve_save_states(
                save_only_lora, dtype
            )

        barrier()

        if self.rank == 0:
            # save checkpoint in tmp path
            safetensors.torch.save_file(
                states,
                self.consolidated_path(
                    tmp_dst, save_only_lora=save_only_lora
                ),  # always use safetensors for checkpointing
            )
            self.write_params_info(tmp_dst)
            assert not self.dst_dir.exists(), f"should not happen! {self.dst_dir}"
            tmp_dst.rename(self.dst_dir)

            logger.info(
                f"Done dumping checkpoint in {self.dst_dir} for step: {self.state.step}"
            )

            # 보존 정책: best 모드(점수 기준 top-N) 또는 recency 모드(최근 N개)
            if self.keep_best_metric is not None and score is not None:
                ckpts_to_delete = self.prune_by_best(self.state.step, score)
                logger.info(
                    f"Done deleting checkpoints {', '.join([str(c) for c in ckpts_to_delete])}"
                )
            elif self.num_ckpt_keep is not None:
                ckpts_to_delete = self.delete_old_ckpts()
                logger.info(
                    f"Done deleting checkpoints {', '.join([str(c) for c in ckpts_to_delete])}"
                )

        main_logger_info("Done!")
