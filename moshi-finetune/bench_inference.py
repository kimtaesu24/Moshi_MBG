#!/usr/bin/env python3
"""Run Moshi offline on a Full-Duplex-Bench v1.x directory.

The upstream Moshi offline entry point accepts one wav at a time.  This wrapper
loads the model once, resets its streaming state between examples, and writes
the ``output.wav`` layout expected by Full-Duplex-Bench.
"""

from __future__ import annotations

import argparse
import inspect
import random
from pathlib import Path

import numpy as np
import sphn
import torch
from tqdm import tqdm

from moshi.models import loaders
from moshi.run_inference import InferenceState


V1_TASKS = (
    "candor_pause_handling",
    "synthetic_pause_handling",
    "candor_turn_taking",
    "icc_backchannel",
    "synthetic_user_interruption",
)


def log(level: str, message: str) -> None:
    print(f"[{level.upper()}] {message}", flush=True)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_checkpoint_files(
    checkpoint_dir: Path | None,
    moshi_weight: Path | None,
    lora_weight: Path | None,
    config_path: Path | None,
) -> tuple[Path | None, Path | None, Path | None]:
    """Resolve a moshi-finetune checkpoint without hiding explicit overrides."""
    if checkpoint_dir is None:
        return moshi_weight, lora_weight, config_path

    consolidated = checkpoint_dir / "consolidated"
    if not consolidated.is_dir() and checkpoint_dir.name == "consolidated":
        consolidated = checkpoint_dir
    if not consolidated.is_dir():
        raise FileNotFoundError(
            f"checkpoint has no consolidated directory: {checkpoint_dir}"
        )

    saved_config = consolidated / "config.json"
    saved_lora = consolidated / "lora.safetensors"
    saved_full = consolidated / "consolidated.safetensors"
    config_path = config_path or saved_config
    if saved_lora.is_file():
        lora_weight = lora_weight or saved_lora
    elif saved_full.is_file():
        moshi_weight = moshi_weight or saved_full
    elif lora_weight is None and moshi_weight is None:
        raise FileNotFoundError(
            f"neither lora.safetensors nor consolidated.safetensors exists in {consolidated}"
        )
    if config_path is not None and not config_path.is_file():
        raise FileNotFoundError(f"LM config does not exist: {config_path}")
    return moshi_weight, lora_weight, config_path


def collect_inputs(data_dir: Path, tasks: list[str], overwrite: bool) -> list[Path]:
    inputs: list[Path] = []
    for task in tasks:
        task_dir = data_dir / task
        if not task_dir.is_dir():
            raise FileNotFoundError(
                f"benchmark task directory does not exist: {task_dir}"
            )
        for sample_dir in sorted(task_dir.iterdir(), key=lambda path: path.name):
            input_path = sample_dir / "input.wav"
            if not input_path.is_file():
                continue
            if overwrite or not (sample_dir / "output.wav").is_file():
                inputs.append(input_path)
    return inputs


def read_input(path: Path, sample_rate: int, device: str) -> torch.Tensor:
    pcm, _ = sphn.read(str(path), sample_rate=sample_rate)
    if pcm.ndim == 1:
        pcm = pcm[None, :]
    # Full-Duplex-Bench inputs are mono.  For an accidental stereo input, follow
    # upstream Moshi's convention and use channel zero as the user stream.
    return torch.from_numpy(pcm[:1]).float().unsqueeze(0).to(device=device)


def fit_length(pcm: np.ndarray, target_samples: int) -> np.ndarray:
    pcm = np.asarray(pcm, dtype=np.float32).reshape(-1)
    if pcm.size >= target_samples:
        return pcm[:target_samples]
    return np.pad(pcm, (0, target_samples - pcm.size))


@torch.inference_mode()
def run(args: argparse.Namespace) -> int:
    seed_all(args.seed)
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA was requested but is unavailable: {device}")

    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else None
    moshi_weight, lora_weight, config_path = resolve_checkpoint_files(
        checkpoint_dir,
        Path(args.moshi_weight) if args.moshi_weight else None,
        Path(args.lora_weight) if args.lora_weight else None,
        Path(args.config_path) if args.config_path else None,
    )

    log("info", f"loading checkpoint metadata from {args.hf_repo}")
    checkpoint = loaders.CheckpointInfo.from_hf_repo(
        args.hf_repo,
        moshi_weights=moshi_weight,
        mimi_weights=args.mimi_weight,
        tokenizer=args.tokenizer,
        config_path=config_path,
        lora_weights=lora_weight,
    )
    log("info", f"loading Mimi and Moshi on {device} ({args.dtype})")
    mimi = checkpoint.get_mimi(device=device)
    tokenizer = checkpoint.get_text_tokenizer()
    lm = checkpoint.get_moshi(
        device=device, dtype=dtype, fuse_lora=not args.no_fuse_lora
    )
    lm.eval()

    generation = dict(checkpoint.lm_gen_config)
    generation.update(
        use_sampling=not args.greedy,
        temp=args.temp_audio,
        temp_text=args.temp_text,
        top_k=args.top_k_audio,
        top_k_text=args.top_k_text,
    )
    # Moshi 0.2.4 names the first argument ``model_type``; newer checkouts pass
    # the complete CheckpointInfo. Supporting both keeps this script usable with
    # the package pinned by moshi-finetune as well as the sibling source tree.
    first_state_arg = next(iter(inspect.signature(InferenceState).parameters))
    state_context = (
        checkpoint if first_state_arg == "checkpoint_info" else checkpoint.model_type
    )
    state = InferenceState(
        state_context,
        mimi,
        tokenizer,
        lm,
        batch_size=1,
        cfg_coef=args.cfg_coef,
        device=device,
        **generation,
    )

    inputs = collect_inputs(Path(args.data_dir), args.tasks, args.overwrite)
    log("info", f"processing {len(inputs)} benchmark sample(s)")
    failures = 0
    for input_path in tqdm(inputs, unit="sample"):
        output_path = input_path.parent / "output.wav"
        try:
            # InferenceState uses persistent streaming contexts so the expensive
            # CUDA graphs can be reused, but conversation history must not leak.
            mimi.reset_streaming()
            state.lm_gen.reset_streaming()
            input_pcm = read_input(input_path, mimi.sample_rate, device)
            target_samples = input_pcm.shape[-1]
            outputs = state.run(input_pcm)
            if not outputs:
                output_pcm = np.zeros(target_samples, dtype=np.float32)
            else:
                output_pcm = fit_length(outputs[0][1][0].numpy(), target_samples)
            sphn.write_wav(str(output_path), output_pcm, mimi.sample_rate)
        except Exception as exc:  # keep a long benchmark resumable
            failures += 1
            log("error", f"{input_path}: {exc}")
            if not args.continue_on_error:
                raise

    log("info", f"completed {len(inputs) - failures}/{len(inputs)} samples")
    return failures


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--tasks", nargs="+", default=list(V1_TASKS))
    parser.add_argument(
        "--checkpoint-dir", help="checkpoint_XXXXXX or its consolidated directory"
    )
    parser.add_argument("--hf-repo", default=loaders.DEFAULT_REPO)
    parser.add_argument("--moshi-weight")
    parser.add_argument("--mimi-weight")
    parser.add_argument("--tokenizer")
    parser.add_argument("--config-path")
    parser.add_argument("--lora-weight")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--cfg-coef", type=float, default=1.0)
    parser.add_argument("--temp-audio", type=float, default=0.8)
    parser.add_argument("--temp-text", type=float, default=0.7)
    parser.add_argument("--top-k-audio", type=int, default=250)
    parser.add_argument("--top-k-text", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--no-fuse-lora", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser


if __name__ == "__main__":
    raise SystemExit(1 if run(build_parser().parse_args()) else 0)
