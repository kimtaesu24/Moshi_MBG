import logging
import queue as _queue
import threading
from typing import Any, Iterator

import torch

from .args import DataArgs
from .dataset import build_raw_dataset
from .interleaver import Batch, InterleavedTokenizer

logger = logging.getLogger(__name__)

_SENTINEL = object()

# mimi.encode's transient activation buffer scales with the number of clips it is
# handed — measured ~0.23 GiB per 10s clip on this checkpoint.  Encoding a whole
# 256-clip batch in one call therefore spikes ~59 GiB, on the prefetch stream's
# own allocator pool, to produce a few MB of codes.  That spike was the dominant
# source of VRAM oscillation during training.  Chunking caps it at ~8 GiB.
# Keep this fixed: the chunk size perturbs VQ tie-breaking slightly, so a stable
# value keeps tokenization consistent for the whole run.
_ENCODE_CHUNK = 32


class PrefetchDataLoader:
    """
    Runs a data iterator in a background thread so that mimi encoding,
    file I/O, and tokenization overlap with GPU training on the main thread.

    The background thread enqueues (batch, cuda_event) pairs.  The main thread
    inserts a stream-level wait so the training stream does not consume a batch
    before the prefetch stream has finished producing it — no CPU blocking.
    """

    def __init__(self, iterator: Iterator, buffer_size: int = 2):
        self._it = iterator
        self._q: _queue.Queue = _queue.Queue(maxsize=buffer_size)
        # Capture device before spawning thread (main thread already called set_device)
        self._device = torch.cuda.current_device()
        # Dedicated side-stream so prefetch kernels (mimi.encode) run concurrently
        # with the training stream on the main thread.
        self._stream = torch.cuda.Stream(device=self._device)
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self) -> None:
        torch.cuda.set_device(self._device)
        try:
            while True:
                # All CUDA work inside next() (mimi.encode, tensor creation) goes
                # onto self._stream, freeing the main training stream.
                with torch.cuda.stream(self._stream):
                    batch = next(self._it)
                event = torch.cuda.Event()
                event.record(self._stream)
                self._q.put((batch, event))
        except StopIteration:
            self._q.put(_SENTINEL)
        except Exception as exc:
            self._q.put(exc)

    def __iter__(self) -> "PrefetchDataLoader":
        return self

    def __next__(self) -> Any:
        item = self._q.get()
        if item is _SENTINEL:
            raise StopIteration
        if isinstance(item, Exception):
            raise item
        batch, event = item
        # GPU-side wait: training stream will not start until prefetch stream event fires.
        # This is non-blocking for the CPU.
        torch.cuda.current_stream().wait_event(event)
        return batch


def build_data_loader(
    instruct_tokenizer: InterleavedTokenizer,
    args: DataArgs,
    batch_size: int,
    seed: int | None,
    rank: int,
    world_size: int,
    is_eval: bool,
) -> Iterator[Batch]:
    if is_eval:
        assert args.eval_data != "", "No eval data provided."
    pretrain_data = args.train_data if not is_eval else args.eval_data

    # Raw iterator: yields (wav, start_sec, path, vp_emb) with NO mimi.encode.
    raw_dataset = build_raw_dataset(
        pretrain_data=pretrain_data,
        instruct_tokenizer=instruct_tokenizer,
        seed=seed,
        rank=rank,
        world_size=world_size,
        is_eval=is_eval,
        shuffle_pretrain=args.shuffle,
    )

    mimi = instruct_tokenizer.mimi

    _flame_filter = instruct_tokenizer.flame_root  # non-empty → face gen active

    def _batch_generator():
        raw_buf = []
        n_flame_skipped = 0
        for item in raw_dataset:
            # ── Pre-filter: drop audio files with no matching FLAME .npy ─────
            # Collate uses all() — one missing file zeros the face loss for the
            # entire batch.  Check here (cached after first lookup) so we never
            # waste a mimi.encode call on a sample that will ruin the batch.
            if _flame_filter:
                _, _, path, _ = item
                if not instruct_tokenizer.has_flame_file(path):
                    n_flame_skipped += 1
                    if n_flame_skipped % 1000 == 1:
                        logger.warning(
                            f"[data_loader] Skipped {n_flame_skipped} sample(s) so far "
                            f"with no matching FLAME pair (last: {path})."
                        )
                    continue

            raw_buf.append(item)
            if len(raw_buf) < batch_size:
                continue

            # ── Batched mimi encoding ─────────────────────────────────────
            # Prepare: wav [C, T] or [T] → [C, 1, T]; cat all into [C*B, 1, T],
            # then encode in fixed-size chunks (see _ENCODE_CHUNK).
            wav_tensors = []
            for wav, *_ in raw_buf:
                w = torch.as_tensor(wav, dtype=torch.float32, device="cuda")
                if w.dim() == 1:
                    w = w.unsqueeze(0)      # mono [T] → [1, T]
                wav_tensors.append(w)

            C = wav_tensors[0].shape[0]    # channels per sample (typically 2)
            B = len(raw_buf)

            # Record actual audio lengths before zero-padding so we can build
            # validity masks that exclude silence-padded frames from all losses.
            actual_wav_lens = [w.shape[-1] for w in wav_tensors]

            # Last segments of audio files are shorter than full duration (sphn slices
            # to unpadded_len).  Pad all wavs to the same T so torch.cat can proceed.
            max_T = max(w.shape[-1] for w in wav_tensors)
            wav_tensors = [
                torch.nn.functional.pad(w, (0, max_T - w.shape[-1]))
                if w.shape[-1] < max_T else w
                for w in wav_tensors
            ]

            # Stack: [B, C, T] → [B*C, 1, T]
            stacked = torch.cat([w[:, None] for w in wav_tensors], dim=0)  # [B*C, 1, T]

            with torch.no_grad():
                all_tokens = torch.cat(
                    [
                        mimi.encode(stacked[i:i + _ENCODE_CHUNK])
                        for i in range(0, stacked.shape[0], _ENCODE_CHUNK)
                    ],
                    dim=0,
                )  # [B*C, K, T_enc]

            # Reshape to [B, C, K, T_enc] so sample i gets all_tokens[i]: [C, K, T_enc]
            all_tokens = all_tokens.view(B, C, *all_tokens.shape[1:])

            samples = [
                instruct_tokenizer.tokenize_with_encoded_audio(
                    all_tokens[i], start_sec, path, vp_emb,
                    actual_wav_samples=actual_wav_lens[i],
                )
                for i, (_, start_sec, path, vp_emb) in enumerate(raw_buf)
            ]

            # ── Post-tokenization safety check ────────────────────────────
            # Pre-filter handles ordinary missing pairs. This catches files that
            # become unavailable plus malformed/out-of-range targets rejected by
            # _load_face_motion; that method logs the precise rejection reason.
            if _flame_filter:
                valid_samples = [s for s in samples if s.face_motion_gt is not None]
                n_oob = len(samples) - len(valid_samples)
                if n_oob > 0:
                    logger.warning(
                        f"[data_loader] Dropping {n_oob}/{len(samples)} sample(s) "
                        "because FLAME targets were unavailable or invalid; see "
                        "the preceding [FLAME] warning for details."
                    )
                    samples = valid_samples
                if not samples:
                    raw_buf = []
                    continue

            assert all(s.codes.dim() == 3 and len(s.codes) == 1 for s in samples)
            yield Batch.collate(samples)
            raw_buf = []

    raw = _batch_generator()
    return PrefetchDataLoader(raw, buffer_size=2)
