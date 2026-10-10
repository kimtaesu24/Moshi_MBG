# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from os.path import isfile, splitext
import logging
import numpy as np
import sys
from typing import Optional, Union, List, Tuple, Callable, Iterator
import sphn
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from ..utils.sampling import sample_token
from ..utils.compile import CUDAGraphed
from ..modules.streaming import StreamingStateDict, StreamingContainer, StreamingModule, load_streaming_state
from ..modules.transformer import (
    StreamingTransformer,
    create_norm_fn,
)
from .backchannel_vap import BackchannelModule
from .vap_gpt_module import VapGPTBackchannelModule, BackchannelOutput2

logger = logging.getLogger(__name__)

AUDIO_TOKENS_PER_STREAM = 8
FRAME_RATE_HZ = 12.5
SILENCE_TOKENS = np.array([948, 243, 1178, 546, 1736, 1030, 1978, 2008], dtype=np.int64)
SINE_TOKENS    = np.array([430, 1268, 381, 1611, 1095, 1495, 56, 472], dtype=np.int64)


@dataclass
class LMOutput:
    # The logits are already re-aligned with the input codes
    # hence no extra shift is required, e.g. when computing CE
    logits: torch.Tensor  # [B, K, T, card]
    mask: torch.Tensor  # [B, K, T]
    text_logits: torch.Tensor  # [B, 1, T, text_card]
    text_mask: torch.Tensor  # [B, 1, T]
    vap_logits: Optional[torch.Tensor] = None          # [B, T, 256] backchannel VAP logits
    commitment_loss: Optional[torch.Tensor] = None    # scalar — Alt 3 commitment loss
    face_pred: Optional[torch.Tensor] = None           # [B, T_face, 54] teacher-forced face motion
    face_outputs: Optional[dict] = None                # full CausalSoftVQContinuousTransformer output dict
    bc_stats: Optional[dict] = None                    # scalar tensors: y_bc_mean, s_pad_mean, g_soft_mean, g_final_rate
    bc_logits: Optional[torch.Tensor] = None  # [B, T, 3] — future text-slot class logits (0=PAD, 1=EPAD, 2=WORD)
    vad_logits: Optional[torch.Tensor] = None  # [B, T, 2] — CURRENT-frame VA logits (0=user, 1=agent), BCE-supervised
    transformer_out: Optional[torch.Tensor] = None  # [B, T, D], for online-teacher KD
    fusion_residual: Optional[torch.Tensor] = None  # [B, T], learned EPAD residual
    decision_logits: Optional[torch.Tensor] = None  # [B, T], decision-head evidence (if enabled)


def _delay_sequence(delays: List[int], tensor: torch.Tensor, padding: torch.Tensor) -> torch.Tensor:
    B, K, T = tensor.shape
    assert len(delays) == K, (len(delays), K)
    outs = []

    for k, delay in enumerate(delays):
        assert delay >= 0
        line = tensor[:, k].roll(delay, dims=1)
        if delay > 0:
            line[:, :delay] = padding[:, k]
        outs.append(line)
    return torch.stack(outs, dim=1)


def _undelay_sequence(delays: List[int], tensor: torch.Tensor,
                      fill_value: Union[int, float] = float('NaN')) -> Tuple[torch.Tensor, torch.Tensor]:
    B, K, T, *_ = tensor.shape
    assert len(delays) == K
    mask = torch.ones(B, K, T, dtype=torch.bool, device=tensor.device)
    outs = []
    if all([delay == 0 for delay in delays]):
        return tensor, mask
    for k, delay in enumerate(delays):
        assert delay >= 0
        line = tensor[:, k].roll(-delay, dims=1)
        if delay > 0:
            line[:, -delay:] = fill_value
            mask[:, k, -delay:] = 0
        outs.append(line)
    return torch.stack(outs, dim=1), mask


def create_sinewave(duration: float, sample_rate: int) -> np.ndarray:
    """Return a 440 Hz 'silent' sinewave of the given duration."""
    t = np.linspace(0.0, duration, int(sample_rate * duration), endpoint=False)
    amplitude = 0.5
    return amplitude * np.sin(2 * np.pi * 440.0 * t).astype(np.float32)


def normalize_audio(wav: np.ndarray, sr: int, target_lufs: float) -> np.ndarray:
    """Normalize **mono** audio to a target LUFS level."""
    import pyloudnorm as pyln
    # Ensure shape is (T,)
    if wav.ndim == 2 and wav.shape[0] == 1:
        wav = wav[0]

    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(wav)
    return pyln.normalize.loudness(wav, loudness, target_lufs)


def load_audio(
    filepath: str, sample_rate: int, 
):
    """Yields audio samples in intervals of sample_interval_size"""
    sample_pcm, sample_sr = sphn.read(filepath)
    sample_pcm = sphn.resample(
        sample_pcm, src_sample_rate=sample_sr, dst_sample_rate=sample_rate
    )  # shape: (C, T)
    return sample_pcm

def _iterate_audio(sample_pcm, sample_interval_size, max_len=sys.maxsize, pad=True):
    cnt = 0
    while sample_pcm.shape[-1] > 0 and cnt < max_len:
        sample = sample_pcm[:, :sample_interval_size]
        sample_pcm = sample_pcm[:, sample_interval_size:]
        if sample_pcm.shape[-1] == 0 and pad:
            sample = np.concatenate(
                [
                    sample,
                    np.zeros(
                        (
                            sample.shape[0],
                            sample_interval_size - sample.shape[-1],
                        )
                    ),
                ],
                axis=1,
            )
        cnt += 1
        yield sample[0:1]  # shape: (1, T)


def encode_from_sphn(mimi, samples, max_batch=sys.maxsize):
    """
    Takes an iterator of samples, batches them, encodes them;
    and yields the encoded samples one sample at a time in the same order.
    """
    device = next(mimi.parameters()).device
    current_batch = []
    done_flag = False
    # TO-DO: Fix the batching bug
    max_batch = 1

    while True:
        try:
            sample = next(samples)
            tensor = torch.tensor(sample, dtype=torch.float32, device=device)
            tensor = tensor.unsqueeze(0)  # shape: (1, C, T)                                                                                                      
            current_batch.append(tensor)
        except StopIteration:
            done_flag = True

        if (not done_flag) and len(current_batch) < max_batch:
            continue
        if not current_batch:
            break

        batch = torch.cat(current_batch, dim=0)  # shape: (B, C, T)
        encoded = mimi.encode(batch)  # shape: (B, K, F)
        separated = torch.unbind(encoded, dim=0)  # shape: (K, F)
        reshaped = [x.unsqueeze(0) for x in separated]  # shape: (1, K, F)
        detached = [x.detach().clone() for x in reshaped]

        current_batch = []
        yield from detached  # shape: (1, K, F)

        if done_flag:
            break


class ScaledEmbedding(torch.nn.Embedding):
    """Boost learning rate for embeddings (with `scale`).

    Args:
        norm (bool): if True, uses a layer norm after the embedding.
        zero_idx (int): special value indicating that the output should be exactly 0.
    """

    def __init__(self, *args, norm: bool = False, zero_idx: int = -1, **kwargs):
        super().__init__(*args, **kwargs)
        self.norm = None
        if norm:
            self.norm = create_norm_fn("layer_norm", self.embedding_dim)
        assert zero_idx < 0, "Please use negative values for the zero_idx."
        self.zero_idx = zero_idx

    def forward(self, input, *args, **kwargs):
        is_zero = input == self.zero_idx
        zero = torch.zeros(1, dtype=input.dtype, device=input.device)
        input = input.clamp(min=0)
        y = super().forward(input, *args, **kwargs)
        if self.norm is not None:
            y = self.norm(y)
        y = torch.where(is_zero[..., None], zero, y)
        return y


class LMModel(StreamingContainer):
    """Transformer-based language model on multiple streams of codes.

    Args:
        n_q (int): Number of parallel streams to model as input.
        dep_q (int): Number of parallel streams to model in the depformer.
        card (int): Cardinality, vocabulary size.
        text_card (int): Cardinality of the text vocabulary.
        dim (int): Dimension of the transformer encoder.
        num_heads (int): Number of heads for the transformer encoder.
        hidden_scale (int): Scale for hidden feed forward dimension of the transformer encoder.
        norm (str): Normalization method.
        norm_emb (bool): Whether to normalize embeddings.
        bias_proj (bool): Use bias for output projections.
        depformer_*: params used for the Depformer Transformer, all the other will be shared.
        depformer_multi_linear (bool): if True, uses one linear layer per codebook to project the
            output of the main transformer to the Depformer latent space.
        depformer_dim_feedforward (int| list[int]| None): If None, defaults to hidden_scale * depformer_dim.
        existing_text_padding_id (bool): if True, will use a different token for the initial text token, and
            the text padding token.
        same_initial (bool): if True, uses the same initial tokens for both text and audio mode.
        **kwargs: Additional parameters for the transformer encoder.
    """

    def __init__(
        self,
        delays: List[int] = [0],
        n_q: int = 8,
        dep_q: int = 8,
        card: int = 1024,
        text_card: int = 32000,
        dim: int = 128,
        num_heads: int = 8,
        hidden_scale: int = 4,
        norm: str = "layer_norm",
        norm_emb: bool = False,
        bias_proj: bool = False,
        depformer_dim: int = 256,
        depformer_dim_feedforward: int | list[int] | None = None,
        depformer_multi_linear: bool = False,
        depformer_weights_per_step: bool = False,
        depformer_weights_per_step_schedule: list[int] | None = None,
        depformer_pos_emb: str = "sin",
        existing_text_padding_id: Optional[int] = None,
        context: Optional[int] = None,
        # ── Backchannel VAP ──────────────────────────────────────────
        backchannel_enabled: bool = False,
        backchannel_module_type: str = "mlp",  # "mlp" | "vap_gpt"
        backchannel_vap_dim: int = 256,
        backchannel_bc_hidden: int = 512,
        backchannel_pad_token_id: Optional[int] = None,
        backchannel_epad_token_id: Optional[int] = None,
        backchannel_gumbel_temp_init: float = 1.0,
        backchannel_gumbel_temp_min: float = 0.5,
        backchannel_gumbel_anneal_rate: float = 0.0001,
        backchannel_fusion_trainable: bool = False,
        backchannel_fusion_bc_init: float = 0.0,
        backchannel_fusion_vap_init: float = 0.0,
        backchannel_fusion_vad_init: float = 0.0,
        backchannel_fusion_bias_init: float = 0.0,
        # Which listener outputs enter the EPAD residual: any of bc, vap, vad, decision.
        backchannel_fusion_evidence: str = "bc,vap,vad",
        # Learned scale a on the backbone's own EPAD-vs-PAD log-odds: residual += (a-1)*s.
        backchannel_fusion_calibration: bool = False,
        backchannel_fusion_calibration_init: float = 1.0,
        # Decision head (evidence trained only by the fused EPAD loss) and whether the
        # auxiliary VAP/VAD/BC losses may update the shared listener trunk.
        backchannel_decision_head: bool = False,
        backchannel_decision_hidden: int = 128,
        backchannel_aux_to_trunk: str = "full",
        # > 0: no EPAD residual on frames where the agent emitted a non-PAD text token
        # within this many previous frames (it is mid-utterance). Applied identically in
        # training (from the teacher-forced text stream) and in generation (LMGen).
        backchannel_fusion_gate_frames: int = 0,
        # VapGPT-specific params (used when backchannel_module_type == "vap_gpt")
        backchannel_vap_repo_path: str = "",
        backchannel_vap_checkpoint: Optional[str] = None,
        backchannel_vap_channel_layers: int = 1,
        backchannel_vap_cross_layers: int = 3,
        backchannel_vap_num_heads: int = 4,
        backchannel_vap_dropout: float = 0.1,
        backchannel_vap_use_silence_ctx_proj: bool = True,
        # ── Face Generation Module ────────────────────────────────────────────
        face_module_enabled: bool = False,
        face_module_dir: str = "",
        face_module_checkpoint: Optional[str] = None,
        face_module_hidden_dim: int = 512,
        face_module_layers: int = 6,
        face_module_heads: int = 8,
        face_module_code_dim: int = 32,
        face_module_prior_warmup_frames: int = 10,
        face_module_version: int = 1,
        face_module_ss_prob: float = 0.0,
        face_module_ss_ramp_steps: int = 0,
        face_module_ss_start_step: int = 0,
        face_module_ss_keep_head_frames: int = 0,
        face_module_detach_llm_features: bool = False,
        # ── Mimi Model for internal audio decoding (server) ──────────────────
        # Used by the face module to decode agent audio from predicted logits.
        mimi_enabled: bool = False,
        mimi_checkpoint: Optional[str] = None,
        device=None,
        dtype=None,
        **kwargs,
    ):
        super().__init__()
        self.n_q = n_q
        self.dep_q = dep_q
        self.card = card
        self.text_card = text_card
        self.face_module_ss_prob = float(face_module_ss_prob)
        self.face_module_ss_ramp_steps = int(face_module_ss_ramp_steps)
        self.face_module_ss_start_step = int(face_module_ss_start_step)
        self.face_module_ss_keep_head_frames = int(face_module_ss_keep_head_frames)
        self.face_module_detach_llm_features = bool(
            face_module_detach_llm_features
        )
        self.backchannel_fusion_trainable = bool(backchannel_fusion_trainable)
        # Shape (1,) rather than a 0-dim scalar: FSDP's flat-param sharding
        # rejects scalar parameters outright.
        self.backchannel_fusion_bc_weight = torch.nn.Parameter(
            torch.tensor([float(backchannel_fusion_bc_init)]),
            requires_grad=self.backchannel_fusion_trainable,
        )
        self.backchannel_fusion_vap_weight = torch.nn.Parameter(
            torch.tensor([float(backchannel_fusion_vap_init)]),
            requires_grad=self.backchannel_fusion_trainable,
        )
        self.backchannel_fusion_vad_weight = torch.nn.Parameter(
            torch.tensor([float(backchannel_fusion_vad_init)]),
            requires_grad=self.backchannel_fusion_trainable,
        )
        self.backchannel_fusion_bias = torch.nn.Parameter(
            torch.tensor([float(backchannel_fusion_bias_init)]),
            requires_grad=self.backchannel_fusion_trainable,
        )
        self.backchannel_fusion_evidence = frozenset(
            e.strip() for e in str(backchannel_fusion_evidence).split(",") if e.strip())
        unknown = self.backchannel_fusion_evidence - {"bc", "vap", "vad", "decision"}
        if unknown:
            raise ValueError(f"unknown backchannel_fusion_evidence entries: {sorted(unknown)}")
        if "decision" in self.backchannel_fusion_evidence and not backchannel_decision_head:
            raise ValueError("fusion evidence 'decision' needs backchannel_decision_head=True")
        self.backchannel_fusion_gate_frames = int(backchannel_fusion_gate_frames or 0)
        self.backchannel_fusion_calibration = bool(backchannel_fusion_calibration)
        if self.backchannel_fusion_calibration:
            self.backchannel_fusion_implicit_scale = torch.nn.Parameter(
                torch.tensor([float(backchannel_fusion_calibration_init)]),
                requires_grad=self.backchannel_fusion_trainable,
            )
        assert len(delays) == self.num_codebooks, "unexpected number of delays"
        self.delays = delays
        self.dim = dim
        self.existing_text_padding_id = existing_text_padding_id
        self.context = context
        self.depformer_weights_per_step_schedule = depformer_weights_per_step_schedule
        if depformer_weights_per_step_schedule is not None:
            assert len(depformer_weights_per_step_schedule) == dep_q
        kwargs["context"] = context
        EmbeddingFactory = partial(
            ScaledEmbedding,
            norm=norm_emb,
            device=device,
            dtype=dtype,
            zero_idx=self.zero_token_id,
        )
        self.EmbeddingFactory = EmbeddingFactory
        self.emb = torch.nn.ModuleList(
            [EmbeddingFactory(self.card + 1, dim) for _ in range(n_q)]
        )
        # Text card + padding token (if not in the original tokenizer)
        extra_text = self.existing_text_padding_id is None
        # Unlike for audio, here we authorize the model to output the special token.
        self.text_emb = EmbeddingFactory(text_card + 1, dim)
        self.text_linear = torch.nn.Linear(dim, text_card + extra_text, bias=bias_proj)
        depformer_prefix = "depformer_"
        main_kwargs = {
            k: v for k, v in kwargs.items() if not k.startswith(depformer_prefix)
        }
        self.transformer = StreamingTransformer(
            d_model=dim,
            num_heads=num_heads,
            dim_feedforward=int(hidden_scale * dim),
            norm=norm,
            device=device,
            dtype=dtype,
            **main_kwargs,
        )
        self.out_norm = create_norm_fn(norm, dim)
        self.depformer_multi_linear = depformer_multi_linear
        kwargs_dep = main_kwargs.copy()
        kwargs_dep.update(
            {
                k.removeprefix(depformer_prefix): v
                for k, v in kwargs.items()
                if k.startswith(depformer_prefix)
            }
        )
        kwargs_dep["positional_embedding"] = depformer_pos_emb
        kwargs_dep["context"] = None
        if depformer_weights_per_step:
            kwargs_dep["weights_per_step"] = dep_q
        if depformer_multi_linear:
            # One linear layer per codebook to project different informations from the main model.
            self.depformer_in = torch.nn.ModuleList(
                [torch.nn.Linear(dim, depformer_dim, bias=False) for _ in range(dep_q)]
            )
        else:
            self.depformer_in = torch.nn.ModuleList(
                [torch.nn.Linear(dim, depformer_dim, bias=False)]
            )
        # Only using up to dep_q - 1 because the last codebook is never an input to Depformer.
        self.depformer_emb = torch.nn.ModuleList(
            [EmbeddingFactory(self.card + 1, depformer_dim) for _ in range(dep_q - 1)]
        )
        self.depformer_text_emb = EmbeddingFactory(text_card + 1, depformer_dim)
        if depformer_dim_feedforward is None:
            depformer_dim_feedforward = int(hidden_scale * depformer_dim)
        self.depformer = StreamingTransformer(
            d_model=depformer_dim,
            dim_feedforward=depformer_dim_feedforward,
            norm=norm,
            device=device,
            dtype=dtype,
            **kwargs_dep,
        )
        self.depformer.set_streaming_propagate(False)
        dim = depformer_dim  # we will directly apply the next linears to the output of the Depformer.

        self.linears = torch.nn.ModuleList(
            [torch.nn.Linear(dim, self.card, bias=bias_proj) for _ in range(dep_q)]
        )

        # ── Backchannel VAP Module ────────────────────────────────────────
        self.backchannel_enabled = backchannel_enabled
        # Recorded here rather than probed with isinstance() later: FSDP and
        # activation checkpointing replace self.backchannel with a wrapper, so
        # an isinstance() check at call time sees the wrapper and silently
        # reports False.
        self.backchannel_needs_audio_feats = (
            backchannel_enabled and backchannel_module_type == "vap_gpt"
        )
        if backchannel_enabled:
            # Default to the model's actual text PAD/EPAD token IDs when not explicitly set.
            # text_padding_token_id = existing_text_padding_id (= 3 for Personaplex).
            # end_of_text_padding_id = 0 (hardcoded property).
            _pad_id = backchannel_pad_token_id if backchannel_pad_token_id is not None \
                else self.text_padding_token_id
            _epad_id = backchannel_epad_token_id if backchannel_epad_token_id is not None \
                else self.end_of_text_padding_id
            if backchannel_module_type == "vap_gpt":
                self.backchannel = VapGPTBackchannelModule(
                    lm_dim=self.dim,
                    depformer_dim=depformer_dim,
                    card=self.card,
                    vap_repo_path=backchannel_vap_repo_path,
                    checkpoint_path=backchannel_vap_checkpoint,
                    vap_dim=backchannel_vap_dim,
                    channel_layers=backchannel_vap_channel_layers,
                    cross_layers=backchannel_vap_cross_layers,
                    num_heads=backchannel_vap_num_heads,
                    dropout=backchannel_vap_dropout,
                    bc_hidden=backchannel_bc_hidden,
                    use_silence_ctx_proj=backchannel_vap_use_silence_ctx_proj,
                    pad_token_id=_pad_id,
                    epad_token_id=_epad_id,
                    gumbel_temp_init=backchannel_gumbel_temp_init,
                    gumbel_temp_min=backchannel_gumbel_temp_min,
                    gumbel_anneal_rate=backchannel_gumbel_anneal_rate,
                    decision_head=backchannel_decision_head,
                    decision_hidden=backchannel_decision_hidden,
                    aux_to_trunk=backchannel_aux_to_trunk,
                )
            else:  # "mlp"
                self.backchannel = BackchannelModule(
                    lm_dim=self.dim,
                    depformer_dim=depformer_dim,
                    card=self.card,
                    vap_dim=backchannel_vap_dim,
                    bc_hidden=backchannel_bc_hidden,
                    pad_token_id=_pad_id,
                    epad_token_id=_epad_id,
                    gumbel_temp_init=backchannel_gumbel_temp_init,
                    gumbel_temp_min=backchannel_gumbel_temp_min,
                    gumbel_anneal_rate=backchannel_gumbel_anneal_rate,
                )
        else:
            self.backchannel = None

        # ── Face Generation Module ────────────────────────────────────────────
        self.face_module = None
        if face_module_enabled and not face_module_dir:
            raise ValueError(
                "face_module_enabled=True requires a non-empty face_module_dir"
            )
        if face_module_enabled and face_module_dir:
            if face_module_dir not in sys.path:
                sys.path.insert(0, face_module_dir)
            if face_module_checkpoint is None:
                raise ValueError(
                    "face_module_enabled=True requires face_module_checkpoint; "
                    "refusing to train an uninitialised face module."
                )
            if not isfile(face_module_checkpoint):
                raise FileNotFoundError(
                    f"Required face checkpoint does not exist: {face_module_checkpoint}"
                )
            try:
                # Version switch: v2 (softvq_continuous_online_train_v2) adds blink
                # modelling, block-causal audio chunks and MTP look-ahead over v1.
                # v6 additionally adds partner-context cross-attention (partner_layers,
                # use_vap). v7 drops that v6-only feature and otherwise refines the
                # v2-era architecture — its constructor matches v2's, not v6's.
                # Checkpoints from different versions are NOT interchangeable —
                # face_module_version must match the checkpoint's training script exactly.
                if int(face_module_version) >= 7:
                    from softvq_continuous_online_train_v7 import CausalSoftVQContinuousTransformer as _FaceModel  # noqa: PLC0415
                elif int(face_module_version) == 6:
                    from softvq_continuous_online_train_v6 import CausalSoftVQContinuousTransformer as _FaceModel  # noqa: PLC0415
                elif int(face_module_version) >= 2:
                    from softvq_continuous_online_train_v2 import CausalSoftVQContinuousTransformer as _FaceModel  # noqa: PLC0415
                else:
                    from softvq_continuous_online_train import CausalSoftVQContinuousTransformer as _FaceModel  # noqa: PLC0415
                # Read architecture hyperparams from checkpoint when available,
                # falling back to explicit constructor arguments.
                ckpt_args: dict = {}
                if face_module_checkpoint is not None:
                    _raw = torch.load(face_module_checkpoint, map_location="cpu", weights_only=False)
                    ckpt_args = _raw.get("args", {})
                _face_kwargs = dict(
                    hidden_dim=int(ckpt_args.get("hidden_dim", face_module_hidden_dim)),
                    layers=int(ckpt_args.get("layers", face_module_layers)),
                    heads=int(ckpt_args.get("heads", face_module_heads)),
                    code_dim=int(ckpt_args.get("code_dim", face_module_code_dim)),
                    prior_warmup_frames=int(ckpt_args.get("prior_warmup_frames", face_module_prior_warmup_frames)),
                )
                if int(face_module_version) >= 2:
                    # v2+ arch args — MUST mirror the checkpoint: the MTP/look-ahead
                    # modules are only instantiated when lookahead_frames > 0, so a
                    # mismatch breaks the (strict) state-dict load below.
                    _face_kwargs.update(
                        lookahead_frames=int(ckpt_args.get("lookahead_frames", 0)),
                        chunk_frames=int(ckpt_args.get("chunk_frames", 1)),
                        token_vocab=int(ckpt_args.get("token_vocab", 2048)),
                        la_temp=float(ckpt_args.get("la_temp", 0.5)),
                    )
                if int(face_module_version) == 6:
                    # v6-only arch args (partner-context cross-attention). Not present
                    # in v7 — its constructor reverted to the v2 parameter set.
                    _face_kwargs.update(
                        partner_layers=int(ckpt_args.get("partner_layers", 2)),
                        use_vap=bool(ckpt_args.get("use_vap", True)),
                    )
                face_net = _FaceModel(**_face_kwargs)
                if "model" not in _raw:
                    raise KeyError(
                        f"Face checkpoint has no 'model' state dict: {face_module_checkpoint}"
                    )
                face_net.load_state_dict(_raw["model"], strict=True)
                # Reset only the newly connected v7 LLM-feature residual after
                # loading the standalone face checkpoint. This guarantees that
                # end-to-end step 0 reproduces the pretrained audio-only model.
                if int(face_module_version) >= 7:
                    if not hasattr(face_net, "llm_proj"):
                        raise AttributeError("v7 face model is missing required llm_proj")
                    torch.nn.init.zeros_(face_net.llm_proj.weight)
                    if face_net.llm_proj.bias is not None:
                        torch.nn.init.zeros_(face_net.llm_proj.bias)
                    logger.info("[LMModel] v7 llm_proj zero-initialized after checkpoint load.")
                logger.info(f"[LMModel] Face module loaded from: {face_module_checkpoint}")
                self.face_module = face_net
                logger.info("[LMModel] Face generation module initialized as submodule.")
            except Exception as _e:
                raise RuntimeError(
                    f"Face module initialization failed for {face_module_checkpoint}; "
                    "training cannot continue without the requested face branch."
                ) from _e

        # ── Internal Mimi for face audio decoding (server) ──────────────────
        # Each rank holds a frozen full copy (small model — no FSDP needed).
        # Used to decode the agent's predicted audio codes into Mimi latents
        # so the face module can run in "generated audio" mode.
        self.mimi = None
        if mimi_checkpoint is not None:
            try:
                from .loaders import get_mimi  # local import to avoid circular dependency
                self.mimi = get_mimi(mimi_checkpoint, device=device)
                self.mimi.eval()
                for p in self.mimi.parameters():
                    p.requires_grad = False
                logger.info(f"[LMModel] Internal Mimi loaded from: {mimi_checkpoint}")
            except Exception as _e:
                logger.warning(
                    f"[LMModel] Mimi initialization failed: {_e}. "
                    "Internal face audio decoding will not be available."
                )

    @property
    def initial_token_id(self) -> int:
        """Token id for the start of sequence (audio)."""
        return self.card

    @property
    def text_initial_token_id(self) -> int:
        """Token id for the start of sequence (text)."""
        return self.text_card

    @property
    def text_padding_token_id(self) -> int:
        """Token id for text padding."""
        if self.existing_text_padding_id is None:
            return self.text_card
        else:
            return self.existing_text_padding_id

    @property
    def end_of_text_padding_id(self) -> int:
        """Token id for optionally marking the last padding step for a word."""
        return 0

    @property
    def zero_token_id(self) -> int:
        """Special value in the input tokens, indicating that no sampling should
        happen for that value, and no input should be given to the model."""
        return -1

    @property
    def ungenerated_token_id(self) -> int:
        """Special value that can be provided in the prompt to indicate that this specific
        value should be predicted and sampled. This allows for partial teacher forcing, by generating
        one modality, with the other one fixed.
        """
        return -2

    @property
    def device(self):
        first_param = next(iter(self.parameters()))
        return first_param.device

    @property
    def num_codebooks(self) -> int:
        return self.n_q + 1

    @property
    def num_audio_codebooks(self) -> int:
        return self.n_q

    @property
    def audio_offset(self) -> int:
        return 1

    def _get_initial_token(self) -> torch.Tensor:
        # Returns the initial token that will be fed to the model to predict the very first timestep.
        # The output shape will be [B, K, 1].
        device = next(iter(self.parameters())).device
        zero = torch.full(
            [1, 1, 1], self.zero_token_id, device=device, dtype=torch.long
        )
        special = torch.full_like(zero, self.initial_token_id)

        text_special = torch.full_like(zero, self.text_initial_token_id)
        audio_token = special
        text_token = text_special
        audio_token = audio_token.expand(-1, self.num_audio_codebooks, -1)
        token = torch.cat([text_token, audio_token], dim=1)
        return token
    
    def embed_codes(self, sequence: torch.Tensor) -> torch.Tensor:
        B, K, S = sequence.shape
        assert (
            K == self.num_codebooks
        ), f"Sequence shape {sequence.shape} must match the number of codebooks."
        input_sequence = sequence
        input_ = None
        for cb_index in range(self.num_audio_codebooks):
            audio_emb = self.emb[cb_index](
                input_sequence[:, cb_index + self.audio_offset]
            )
            input_ = audio_emb if input_ is None else input_ + audio_emb
        text_emb = self.text_emb(input_sequence[:, 0])
        input_ = text_emb if input_ is None else input_ + text_emb
        return input_

    # ── EPAD fusion (shared by training forward and streaming generation) ──
    def fusion_evidences(self, bc_logits, vap_logits, vad_logits) -> dict:
        """Per-frame listener evidences as log-odds (fp32), any leading shape."""
        bc = bc_logits.float()
        start = bc[..., 1] - torch.logsumexp(bc[..., [0, 2]], dim=-1)  # START vs (PAD ∪ WORD)
        vap = torch.zeros_like(start)
        if vap_logits is not None:  # P(agent speaks in the nearest VAP bin), marginalised
            v = vap_logits.float()
            near = (torch.arange(v.shape[-1], device=v.device) & 0x08) != 0
            vap = torch.logsumexp(v[..., near], dim=-1) - torch.logsumexp(v[..., ~near], dim=-1)
        user_quiet = -vad_logits[..., 0].float()  # vad_logits[..., 0] = user-active log-odds
        return {"bc": start, "vap": vap, "vad": user_quiet}

    def fusion_residual(self, evidences: dict, implicit: torch.Tensor, decision_logits=None) -> torch.Tensor:
        """Residual added to the EPAD logit. implicit = backbone ℓ[EPAD] − ℓ[PAD] (fp32)."""
        ev_set = self.backchannel_fusion_evidence
        r = self.backchannel_fusion_bias.float().expand_as(implicit)
        if "bc" in ev_set:
            r = r + self.backchannel_fusion_bc_weight.float() * evidences["bc"]
        if "vap" in ev_set:
            r = r + self.backchannel_fusion_vap_weight.float() * evidences["vap"]
        if "vad" in ev_set:
            r = r + self.backchannel_fusion_vad_weight.float() * evidences["vad"]
        if "decision" in ev_set and decision_logits is not None:
            r = r + decision_logits.float()
        if self.backchannel_fusion_calibration:
            r = r + (self.backchannel_fusion_implicit_scale.float() - 1.0) * implicit
        return r

    def fusion_speaking_mask(self, text_tokens: torch.Tensor, frames: int) -> torch.Tensor:
        """[B, T] bool: a non-PAD text token occurs in positions t-frames .. t-1.

        Same rule as LMGen's frame counter (EPAD and words count as speaking; PAD and the
        zero/ungenerated token do not), evaluated on a teacher-forced text stream.
        """
        spoke = ((text_tokens != self.text_padding_token_id)
                 & (text_tokens != self.zero_token_id) & (text_tokens >= 0)).to(torch.int32)
        T = spoke.shape[-1]
        before = F.pad(spoke.cumsum(-1), (1, 0))[..., :T]          # tokens in 0 .. t-1
        before_window = F.pad(before, (frames, 0))[..., :T]         # tokens in 0 .. t-frames-1
        return (before - before_window) > 0

    def forward_codes(
        self,
        sequence: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward_embeddings(self.embed_codes(sequence))
    
    def forward_embeddings(self, input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # print("EMBED:", input[0, 0, :10].float().cpu().tolist()) # DEBUG
        transformer_out = self.transformer(input)
        if self.out_norm:
            transformer_out = self.out_norm(transformer_out)
        assert isinstance(transformer_out, torch.Tensor)
        text_logits = self.text_linear(transformer_out)
        text_logits = text_logits[:, None]
        return transformer_out, text_logits

    def forward_depformer(
        self,
        depformer_cb_index: int,
        sequence: torch.Tensor,
        transformer_out: torch.Tensor,
    ) -> torch.Tensor:
        B, K, S = sequence.shape
        assert (
            K == 1
        ), f"Codebooks for Depformer streaming should be passed 1 by 1, got {K}."
        assert (
            S == 1
        ), f"Steps for Depformer streaming should be passed 1 by 1, got {S}."
        assert (
            transformer_out.shape[1] == 1
        ), "Transformer out should be a for a single step."
        last_token_input: Optional[torch.Tensor] = None
        depformer_input = transformer_out
        if self.depformer_multi_linear:
            depformer_input = self.depformer_in[depformer_cb_index](depformer_input)
        else:
            depformer_input = self.depformer_in[0](depformer_input)
        if depformer_cb_index == 0:
            last_token_input = self.depformer_text_emb(sequence[:, 0])
        else:
            last_token_input = self.depformer_emb[depformer_cb_index - 1](
                sequence[:, 0]
            )
        depformer_input = depformer_input + last_token_input
        assert depformer_input.shape[1] == 1
        # depformer_input is [B, 1, depformer_dim].
        # The streaming state of the depformer ensures that the proper layer is run.
        dep_output = self.depformer(depformer_input)
        logits = self.linears[depformer_cb_index](dep_output)
        logits = logits[:, None]
        assert logits.dim() == 4, logits.shape  # [B, Ka, S, card]
        return logits

    def forward_depformer_training(
        self,
        sequence: torch.Tensor,
        transformer_out: torch.Tensor,
        text_token_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run the DepthTransformer for all codebooks in parallel (training mode).

        Args:
            sequence: [B, K, T] delayed target codes.
            transformer_out: [B, T, dim] main transformer output.
            text_token_emb: [B, T, depformer_dim] optional pre-computed embedding for
                cb_index=0 (the text slot). When provided it replaces the normal
                depformer_text_emb lookup, allowing a differentiable soft embedding
                (e.g. Gumbel-ST interpolation) to be passed directly.
        """
        B, K, T = sequence.shape
        Ka = self.dep_q
        assert (
            K == self.num_codebooks
        ), f"Codebooks for Depformer training should be passed all at once, got {K,}."
        depformer_inputs = []
        for cb_index in range(Ka):
            if self.depformer_multi_linear:
                linear_index = cb_index
                if self.depformer_weights_per_step_schedule is not None:
                    linear_index = self.depformer_weights_per_step_schedule[cb_index]
                transformer_in = self.depformer_in[linear_index](transformer_out)
            else:
                transformer_in = self.depformer_in[0](transformer_out)
            if cb_index == 0:
                # Use provided differentiable embedding if available, otherwise look up.
                token_in = text_token_emb if text_token_emb is not None \
                    else self.depformer_text_emb(sequence[:, 0])
            else:
                token_in = self.depformer_emb[cb_index - 1](sequence[:, cb_index + self.audio_offset - 1])
            depformer_inputs.append(token_in + transformer_in)
        depformer_input = torch.stack(depformer_inputs, 2)
        # depformer_input is [B, T, K, depformer_dim], reshaping to [B * T, K, D]
        depformer_input = depformer_input.view(B * T, Ka, -1)
        depformer_output = self.depformer(depformer_input)
        all_logits = []
        for cb_index in range(Ka):
            logits = self.linears[cb_index](depformer_output[:, cb_index])
            all_logits.append(logits.view(B, T, -1))
        logits = torch.stack(all_logits, 1)
        assert logits.dim() == 4, logits.shape  # [B, Ka, T, card]
        return logits

    def _depformer_first_step_logits(
        self,
        transformer_out: torch.Tensor,
        sequence: torch.Tensor,
    ) -> torch.Tensor:
        """Run DepthTransformer for only the first codebook (cb_index=0)
        and return logits. Used by the backchannel self-reflective gating.

        Args:
            transformer_out: [B, T, dim] from the main transformer.
            sequence: [B, K, T] delayed codes (target side, i.e. delayed_codes[:, :, 1:]).

        Returns:
            logits: [B, T, card] for the first audio codebook.
        """
        B, T, _ = transformer_out.shape
        cb_index = 0
        if self.depformer_multi_linear:
            linear_index = cb_index
            if self.depformer_weights_per_step_schedule is not None:
                linear_index = self.depformer_weights_per_step_schedule[cb_index]
            transformer_in = self.depformer_in[linear_index](transformer_out)
        else:
            transformer_in = self.depformer_in[0](transformer_out)

        # First codebook input: text token embedding
        token_in = self.depformer_text_emb(sequence[:, 0])  # [B, T, depformer_dim]
        depformer_input = (token_in + transformer_in)  # [B, T, depformer_dim]

        # Reshape for depformer: [B*T, 1, depformer_dim]
        depformer_input = depformer_input.view(B * T, 1, -1)
        depformer_output = self.depformer(depformer_input)  # [B*T, 1, depformer_dim]

        logits = self.linears[0](depformer_output[:, 0])  # [B*T, card]
        logits = logits.view(B, T, -1)  # [B, T, card]
        return logits

    def forward(self, codes: torch.Tensor, step: int = 0, voice_prompt_embs: Optional[torch.Tensor] = None,
                audio_feat: Optional[torch.Tensor] = None, gt_face_motion: Optional[torch.Tensor] = None,
                mimi=None, bc_audio_feats: Optional[tuple] = None):
        return self.forward_train(codes, step=step, voice_prompt_embs=voice_prompt_embs,
                                  audio_feat=audio_feat, gt_face_motion=gt_face_motion, mimi=mimi,
                                  bc_audio_feats=bc_audio_feats)

    def forward_teacher(
        self,
        codes: torch.Tensor,
        voice_prompt_embs: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Text/hidden-only forward for a frozen online teacher.

        Skips Depformer, VAP, and face branches to keep online distillation
        substantially cheaper than a second full training forward.
        """
        B = codes.shape[0]
        initial = self._get_initial_token().expand(B, -1, -1)
        delayed_codes = _delay_sequence(self.delays, codes, initial)
        delayed_codes = torch.cat([initial, delayed_codes], dim=2)
        if voice_prompt_embs is not None:
            vp_embs = self.embed_codes(voice_prompt_embs)
            main_embs = self.embed_codes(delayed_codes[:, :, :-1])
            transformer_out, text_logits = self.forward_embeddings(
                torch.cat([vp_embs, main_embs], dim=1)
            )
            prompt_len = vp_embs.shape[1]
            transformer_out = transformer_out[:, prompt_len:]
            text_logits = text_logits[:, :, prompt_len:]
        else:
            transformer_out, text_logits = self.forward_codes(
                delayed_codes[:, :, :-1]
            )
        text_logits, text_mask = _undelay_sequence(
            self.delays[:1], text_logits, fill_value=float("NaN")
        )
        text_mask &= codes[:, :1] != self.zero_token_id
        return text_logits, text_mask, transformer_out

    def forward_train(self, codes: torch.Tensor, step: int = 0, voice_prompt_embs: Optional[torch.Tensor] = None,
                      audio_feat: Optional[torch.Tensor] = None, gt_face_motion: Optional[torch.Tensor] = None,
                      mimi=None, bc_audio_feats: Optional[tuple] = None):  # (agent_audio_feat, user_audio_feat), each [B,T,512]
        B, K, T = codes.shape
        # Delaying codes and removing the last time step that will never be an input.
        initial = self._get_initial_token().expand(B, -1, -1)
        delayed_codes = _delay_sequence(self.delays, codes, initial)
        # Inserting the empty tokens for the first time step.
        delayed_codes = torch.cat([initial, delayed_codes], dim=2)

        # Step 1: LLM Backbone → z_s (transformer_out)
        # If voice_prompt_embs provided, prepend them so the transformer attends to voice context.
        # voice_prompt_embs: [B, K, T_vp] int64 mimi codes — embed with current LM weights each step.
        if voice_prompt_embs is not None:
            vp_embs = self.embed_codes(voice_prompt_embs)  # [B, T_vp, dim]
            main_embs = self.embed_codes(delayed_codes[:, :, :-1])  # [B, T, dim]
            combined_embs = torch.cat([vp_embs, main_embs], dim=1)  # [B, T_vp+T, dim]
            transformer_out, text_logits = self.forward_embeddings(combined_embs)
            V = vp_embs.shape[1]
            transformer_out = transformer_out[:, V:]   # [B, T, dim]
            text_logits = text_logits[:, :, V:]        # [B, 1, T, text_card]
        else:
            transformer_out, text_logits = self.forward_codes(delayed_codes[:, :, :-1])

        # ── Backchannel Module (v2 — pure auxiliary prediction heads) ─────
        vap_logits = None
        vad_logits = None
        commitment_loss = None
        bc_stats = None
        bc_result = None
        if self.backchannel is not None:
            target_codes = delayed_codes[:, :, 1:]
            # Auto-extract per-speaker Mimi latents when not provided externally.
            # This keeps inference self-contained: callers only need to pass mimi.
            # Callers that prepend a text-prompt prefix to `codes` must instead
            # pass bc_audio_feats themselves with the prefix zeroed — decoding
            # the prefix's zero_token_id here yields a real latent, not silence.
            if (bc_audio_feats is None and mimi is not None
                    and self.backchannel_needs_audio_feats):
                with torch.no_grad():
                    _max = self.card - 1
                    _a = mimi.decode_latent(codes[:, 1:9].clamp(0, _max)).transpose(1, 2)
                    _u = mimi.decode_latent(codes[:, 9:17].clamp(0, _max)).transpose(1, 2)
                bc_audio_feats = (
                    _a.to(dtype=transformer_out.dtype),
                    _u.to(dtype=transformer_out.dtype),
                )
            _agent_af = bc_audio_feats[0] if bc_audio_feats is not None else None
            _user_af  = bc_audio_feats[1] if bc_audio_feats is not None else None
            bc_result = self.backchannel(
                transformer_out,
                emb_cb0=self.depformer_text_emb,
                step=step,
                agent_audio_feat=_agent_af,
                user_audio_feat=_user_af,
            )
            vap_logits = bc_result.vap_logits  # [B, T, 256]
            vad_logits = bc_result.vad_logits  # [B, T, 2] — current VA (user, agent)

            # v2 design: NO depformer input conditioning (no bc_embeddings, no gates,
            # no commitment loss). The depformer trains exactly like original Moshi;
            # the backchannel module is a pure auxiliary predictor whose bc/vad/vap
            # heads are CE/BCE-supervised externally (train.py).
            # logits = self.forward_depformer_training(target_codes, transformer_out)

            probs = F.softmax(bc_result.bc_logits, dim=-1)  # [B, T, 3]
            bc_stats = {
                "bc/p_pad_mean":     probs[..., 0].detach().mean(),
                "bc/p_epad_mean":    probs[..., 1].detach().mean(),
                "bc/p_word_mean":    probs[..., 2].detach().mean(),
                "bc/pred_epad_rate": (bc_result.bc_logits.detach().argmax(-1) == 1).float().mean(),
                "bc/vad_user_mean":  torch.sigmoid(bc_result.vad_logits[..., 0]).detach().mean(),
                "bc/vad_agent_mean": torch.sigmoid(bc_result.vad_logits[..., 1]).detach().mean(),
            }
            
        logits = self.forward_depformer_training(delayed_codes[:, :, 1:], transformer_out)

        # map back the logits on pattern sequence to logits on original codes: [B, K, S, card] -> [B, K, T, card]
        # and provide the corresponding mask over invalid positions of tokens. We will with NaN values invalid positions
        # to ensure they properly handled.
        logits, logits_mask = _undelay_sequence(
            self.delays[self.audio_offset:self.audio_offset + self.dep_q],
            logits, fill_value=float('NaN'))
        logits_mask &= (codes[:, self.audio_offset: self.audio_offset + self.dep_q] != self.zero_token_id)
        text_logits, text_logits_mask = _undelay_sequence(self.delays[:1], text_logits, fill_value=float('NaN'))
        text_logits_mask &= (codes[:, :1] != self.zero_token_id)

        # Jointly trained soft state-to-token fusion.  START is compared with
        # both non-onset states (remain silent and already speaking), then the
        # resulting evidence is added only to EPAD.  There is no hard state
        # decision: the regular text CE and sampler retain the final decision.
        fusion_residual = None
        if self.backchannel_fusion_trainable and bc_result is not None:
            # .float() allocates a fresh fp32 tensor (the source is bf16) that nothing
            # else aliases, so the in-place add below is safe (no extra .clone()).
            text_logits = text_logits.float()
            implicit_log_odds = (text_logits[:, 0, :, self.end_of_text_padding_id]
                                 - text_logits[:, 0, :, self.text_padding_token_id])
            evidences = self.fusion_evidences(bc_result.bc_logits, bc_result.vap_logits, bc_result.vad_logits)
            start_log_odds, vap_log_odds, user_quiet_log_odds = evidences["bc"], evidences["vap"], evidences["vad"]
            fusion_residual = self.fusion_residual(evidences, implicit_log_odds, bc_result.decision_logits)
            if self.backchannel_fusion_gate_frames > 0:
                speaking = self.fusion_speaking_mask(codes[:, 0], self.backchannel_fusion_gate_frames)
                fusion_residual = torch.where(speaking, torch.zeros_like(fusion_residual), fusion_residual)
            text_logits[:, 0, :, self.end_of_text_padding_id] += fusion_residual

            if bc_stats is not None:
                bc_stats.update({
                    "fusion/residual_mean": fusion_residual.detach().mean(),
                    "fusion/bc_weight": self.backchannel_fusion_bc_weight.detach().float(),
                    "fusion/vap_weight": self.backchannel_fusion_vap_weight.detach().float(),
                    "fusion/vad_weight": self.backchannel_fusion_vad_weight.detach().float(),
                    "fusion/bias": self.backchannel_fusion_bias.detach().float(),
                })
                if self.backchannel_fusion_calibration:
                    bc_stats["fusion/implicit_scale"] = self.backchannel_fusion_implicit_scale.detach().float()
                if bc_result.decision_logits is not None:
                    bc_stats["fusion/decision_mean"] = bc_result.decision_logits.detach().float().mean()
                    bc_stats["fusion/decision_std"] = bc_result.decision_logits.detach().float().std()
                if getattr(self, "fusion_dynamics_diagnostics", False):
                    with torch.no_grad():
                        valid = text_logits_mask[:, 0].bool().clone()
                        valid[:, :getattr(self, "fusion_dynamics_prompt_length", 0)] = False
                        terms = {"bc": start_log_odds, "vap": vap_log_odds,
                                 "vad": user_quiet_log_odds}
                        weights = {"bc": self.backchannel_fusion_bc_weight,
                                   "vap": self.backchannel_fusion_vap_weight,
                                   "vad": self.backchannel_fusion_vad_weight}
                        bc_stats["fusion/evidence_frames"] = valid.sum().float()
                        for name, term in terms.items():
                            values = term.detach()[valid]
                            coefficient = weights[name].detach().float()
                            for label, data in (("evidence", values), ("contribution", values * coefficient)):
                                bc_stats[f"fusion/{name}_{label}_mean"] = data.mean() if data.numel() else term.new_zeros(())
                                bc_stats[f"fusion/{name}_{label}_std"] = data.std(unbiased=False) if data.numel() else term.new_zeros(())

        # ── Face Generation Module (server logic) ────────────────────────────
        #
        # Design:
        #   • valid_face_batch  — False when gt_face_motion is missing or has zero size
        #                         (can happen when a rank's FLAME file is absent).
        #   • dummy pass         — When valid_face_batch is False we still forward the
        #                         face module with zero tensors so FSDP all-reduce stays
        #                         in sync across all ranks.
        #   • mimi priority      — External mimi (caller) > self.mimi (loaded at init).
        #                         Used to decode agent predicted codes → audio latents.
        #
        face_pred = None
        face_outputs = None

        if self.training and self.face_module is None and gt_face_motion is not None:
            raise RuntimeError(
                "Face ground truth was provided but face_module is unavailable."
            )

        if self.face_module is not None:
            # 1. Determine expected motion dim and whether this batch has valid data.
            valid_face_batch = True
            dummy_B = max(B, 1)

            if gt_face_motion is None:
                if self.training:
                    raise RuntimeError(
                        "Face training requires gt_face_motion for every batch; "
                        "refusing to run a dummy zero-motion pass."
                    )
                valid_face_batch = False
                expected_dim = getattr(
                    getattr(self.face_module, "module", self.face_module),
                    "motion_dim", 54,
                )
            else:
                expected_dim = gt_face_motion.shape[-1]
                if B == 0 or gt_face_motion.shape[0] == 0 or gt_face_motion.shape[1] == 0:
                    valid_face_batch = False
                elif B != gt_face_motion.shape[0]:
                    valid_face_batch = False

            # 2. Resolve which mimi to use and decode agent audio features.
            mimi_to_use = mimi if mimi is not None else self.mimi
            transformer_out_run = (
                transformer_out.detach()
                if self.face_module_detach_llm_features
                else transformer_out
            )
            audio_feat_run = None

            if mimi_to_use is not None:
                # Derive audio features from the model's own predicted audio codes
                # (argmax of depformer logits). NaN padding → treat as code 0 (silence).
                pred_codes = logits.detach().nan_to_num(0.0).argmax(-1).clamp(min=0)  # [B, dep_q, T]
                # dep_q == 16: first 8 are agent codebooks, next 8 are user codebooks.
                # Select agent-only codes matching Mimi's 8 codebooks.
                if pred_codes.shape[1] == 16:
                    agent_codes = pred_codes[:, :8]
                elif pred_codes.shape[1] > 8:
                    agent_codes = pred_codes[:, -8:]
                else:
                    agent_codes = pred_codes
                audio_feat_run = (
                    mimi_to_use.decode_latent(agent_codes)
                    .transpose(1, 2)
                    .to(dtype=transformer_out.dtype)
                )  # [B, T, 512]
            else:
                # Fallback: use externally provided audio_feat (teacher-forced).
                audio_feat_run = audio_feat

            if self.training and audio_feat_run is None:
                raise RuntimeError(
                    "Face training requires either mimi for generated audio features "
                    "or a teacher-forced audio_feat tensor; neither was provided."
                )

            # 3. Build prev_motion for teacher forcing, or create dummy tensors.
            if valid_face_batch:
                start = torch.zeros(
                    B, 1, expected_dim,
                    dtype=transformer_out.dtype, device=transformer_out.device,
                )
                prev_motion = torch.cat([start, gt_face_motion[:, :-1]], dim=1)
            else:
                # Dummy tensors — same device/dtype as transformer_out, minimal B=1.
                T_face = transformer_out_run.shape[1] * 2  # face runs at 2× mimi fps
                prev_motion = torch.zeros(
                    dummy_B, T_face, expected_dim,
                    device=transformer_out_run.device, dtype=transformer_out_run.dtype,
                )
                if audio_feat_run is not None and audio_feat_run.shape[0] != dummy_B:
                    audio_feat_run = audio_feat_run[:1].expand(dummy_B, -1, -1)
                if transformer_out_run.shape[0] != dummy_B:
                    transformer_out_run = transformer_out_run[:1].expand(dummy_B, -1, -1)

            # 4. Forward the face module (always, for FSDP sync).
            if audio_feat_run is not None:
                # Exposure-bias mitigation: use a no-grad teacher-forced pass to
                # replace a scheduled fraction of GT history with self history.
                ss_prob = 0.0
                if self.training and valid_face_batch and self.face_module_ss_prob > 0.0:
                    if step > self.face_module_ss_start_step:
                        if self.face_module_ss_ramp_steps > 0:
                            progress = min(
                                1.0,
                                (step - self.face_module_ss_start_step)
                                / self.face_module_ss_ramp_steps,
                            )
                        else:
                            progress = 1.0
                        ss_prob = self.face_module_ss_prob * progress
                if ss_prob > 0.0:
                    with torch.no_grad():
                        ss_out = self.face_module(
                            audio_feat_run, prev_motion, llm_feat=transformer_out_run
                        )
                        self_prev_motion = torch.cat(
                            [prev_motion[:, :1], ss_out["pred_motion"][:, :-1]], dim=1
                        )
                        ss_mask = (
                            torch.rand(B, prev_motion.shape[1], 1, device=prev_motion.device)
                            < ss_prob
                        )
                        if self.face_module_ss_keep_head_frames > 0:
                            ss_mask[:, :self.face_module_ss_keep_head_frames] = False
                        prev_motion = torch.where(ss_mask, self_prev_motion, prev_motion)
                face_outputs = self.face_module(
                    audio_feat_run, prev_motion, llm_feat=transformer_out_run
                )
                if self.training and face_outputs is None:
                    raise RuntimeError("v7 face module returned None during training")
                if self.training and "pred_motion" not in face_outputs:
                    raise KeyError("v7 face output is missing required 'pred_motion'")
                face_outputs["scheduled_sampling_prob"] = face_outputs["pred_motion"].new_tensor(ss_prob)
                if valid_face_batch:
                    face_pred = face_outputs["pred_motion"]
                else:
                    # Zero placeholder — loss computation downstream will mask it out.
                    T_face_gt = (gt_face_motion.shape[1]
                                 if (gt_face_motion is not None and gt_face_motion.shape[1] > 0)
                                 else transformer_out.shape[1] * 2)
                    face_pred = torch.zeros(
                        (B, T_face_gt, expected_dim),
                        dtype=transformer_out.dtype, device=transformer_out.device,
                    )
                    if face_outputs is not None:
                        face_outputs["is_dummy"] = True

        return LMOutput(
            logits, logits_mask, 
            text_logits, text_logits_mask, 
            vap_logits, commitment_loss,
            face_pred, face_outputs, bc_stats,
            bc_logits=bc_result.bc_logits if bc_result is not None else None,
            vad_logits=vad_logits,
            transformer_out=transformer_out,
            fusion_residual=fusion_residual,
            decision_logits=bc_result.decision_logits if bc_result is not None else None)

@dataclass
class _LMGenState:
    cache: torch.Tensor
    provided: torch.Tensor
    initial: torch.Tensor
    graphed_main: CUDAGraphed
    graphed_embeddings: CUDAGraphed
    graphed_depth: CUDAGraphed
    offset: int = 0

    def reset(self):
        self.offset = 0
        self.provided[:] = False


@torch.no_grad()
def create_loss_report(
    state_cache: torch.Tensor,
    lm_model: LMModel,
    text_logits: torch.Tensor,
    audio_logits: torch.Tensor,
    target: torch.Tensor,
    sampled_text_token: torch.Tensor,
    sampled_audio_tokens: torch.Tensor,
    target_position: int,
) -> dict[str, torch.Tensor]:
    report = {}
    B = state_cache.shape[0]
    # model_tokens is the sampled output from model_logits
    model_tokens = torch.zeros_like(state_cache[:, :, target_position])
    model_tokens[:, 0] = sampled_text_token
    model_tokens[:, 1 : lm_model.dep_q + 1] = sampled_audio_tokens

    report.update(
        {
            "forced_tokens": torch.zeros((B, lm_model.dep_q + 1)),
            "model_tokens": torch.zeros((B, lm_model.dep_q + 1)),
            "ranks_of_forced": torch.zeros((B, lm_model.dep_q + 1)),
            "losses": torch.zeros((B, lm_model.dep_q+1)),
        }
    )
    report["model_tokens"] = model_tokens.clone()
    report["forced_tokens"] = target.clone()

    # Text Channel
    text_logits = text_logits.squeeze(dim=1).squeeze(dim=1)
    target = target[:, 0].squeeze(1).clone()

    text_probs = torch.softmax(text_logits, dim=-1)
    text_ranks = torch.argsort(text_probs, dim=-1, descending=True)
    for b in range(B):
        forced_token = target[b].item()
        try:
            rank = (text_ranks[b] == forced_token).nonzero().item()
        except RuntimeError:
            rank = lm_model.zero_token_id
        report["ranks_of_forced"][b, 0] = rank

    target[target == lm_model.text_initial_token_id] = -100
    text_loss = torch.nn.functional.cross_entropy(
        text_logits,
        target,
        ignore_index=-100,
        )
    report["losses"][:, 0] = text_loss

    # Audio Channels
    for k in range(lm_model.dep_q):
        target = target[:, k+1].squeeze(1).clone()
        channel_logits = audio_logits[:, k, :]

        audio_probs = torch.softmax(channel_logits, dim=-1)
        audio_ranks = torch.argsort(audio_probs, dim=-1, descending=True)
        for b in range(B):
            forced_token = target[b].item()
            try:
                rank = (audio_ranks[b] == forced_token).nonzero().item()
            except RuntimeError:
                rank = lm_model.zero_token_id
            report["ranks_of_forced"][b, k + 1] = rank

        target[target == lm_model.initial_token_id] = -100
        audio_loss = torch.nn.functional.cross_entropy(
            channel_logits,
            target,
            ignore_index=-100,
        )
        report["losses"][:, k + 1] = audio_loss
    return report


class LMGen(StreamingModule[_LMGenState]):
    def __init__(
        self,
        lm_model: LMModel,
        device: str | torch.device,
        use_sampling: bool = True,
        temp: float = 0.8,
        temp_text: float = 0.7,
        top_k: int = 250,
        top_k_text: int = 25,
        check: bool = False,
        report_loss: bool = False,
        return_logits: bool = False,
        audio_silence_frame_cnt: int = 1,
        text_prompt_tokens: Optional[list[int]] = None,
        save_voice_prompt_embeddings: bool = False,
        sample_rate: int = 32000,
        frame_rate: int = FRAME_RATE_HZ,
        mimi=None,
        bc_context_frames: int = 250,
    ):
        assert not lm_model.training, "generation shouldn't be used in training mode."
        super().__init__()

        self.lm_model = lm_model
        self.use_sampling = use_sampling
        self.temp = temp
        self.temp_text = temp_text
        self.top_k = top_k
        self.top_k_text = top_k_text
        self.text_prompt_tokens = text_prompt_tokens
        self.audio_silence_frame_cnt = audio_silence_frame_cnt
        self.voice_prompt = None
        self.zero_text_code = 3
        self._frame_rate = frame_rate
        self._sample_rate = sample_rate
        self._frame_size = int(self._sample_rate / self._frame_rate)
        self._zero_frame = torch.zeros(1, 1, self._frame_size, device=device)
        duration = self._frame_size / self._sample_rate
        sine = create_sinewave(duration, self._sample_rate)
        self._sine_frame = torch.tensor(sine, device=device).unsqueeze(0).unsqueeze(0)  # (1,1,T)
        self.check = check
        self.report_loss = report_loss
        if report_loss:
            return_logits = True
        self.return_logits = return_logits
        self.max_delay = max(
            lm_model.delays
        )  # with delays, we need to generate a few more time steps.
        self.delays_cuda = torch.tensor(
            lm_model.delays, device=lm_model.device, dtype=torch.long
        )
        self.save_voice_prompt_embeddings = save_voice_prompt_embeddings
        self.voice_prompt_audio: Optional[torch.Tensor] = None
        self.voice_prompt_cache: Optional[torch.Tensor] = None
        self.voice_prompt_embeddings: Optional[torch.Tensor] = None
        #self.voice_prompt_mimi_streaming_state: Optional[StreamingStateDict] = None
        self.mimi = mimi  # optional; used to auto-extract bc_audio_feats for VapGPT backchannel
        # backchannel은 있지만 fusion이 학습되지 않은 checkpoint(no_fusion ablation 등)도
        # 그대로 생성할 수 있다. train.py의 LMModel.forward가 fusion_trainable=False일 때
        # EPAD 로짓에 residual을 더하지 않으므로, process_transformer_output도 동일하게
        # residual을 건너뛴다. BC/VAP/VAD 예측 자체는 로깅용으로 계속 계산된다.
        if lm_model.backchannel is not None and not lm_model.backchannel_fusion_trainable:
            logger.info(
                "backchannel_fusion_trainable=False: generating without the fusion "
                "residual, matching the training-time forward path."
            )
        # Rolling-window history for the VapGPT backchannel module: its GPT layers have no
        # KV cache, so at inference we replay the same causal context seen during training
        # by buffering past frames (transformer_out + per-speaker audio feats) up to
        # bc_context_frames, running the module over the whole window, and taking the last
        # (current) frame's decision. Set bc_context_frames to the training sequence length
        # (duration_sec * frame_rate) for exact train/inference parity.
        self.bc_context_frames = bc_context_frames
        self._bc_hist_tout: Optional[torch.Tensor] = None
        self._bc_hist_agent: Optional[torch.Tensor] = None
        self._bc_hist_user: Optional[torch.Tensor] = None
        # Inference-only overrides of the EPAD fusion (analysis of 2026-10-06). The defaults
        # reproduce the trained model exactly; set them after construction.
        #   fusion_gate_frames > 0: add no residual while the agent emitted a non-PAD text
        #       token within the last fusion_gate_frames frames (= it is mid-utterance).
        #       Defaults to the model's backchannel_fusion_gate_frames (the training setting).
        #   fusion_bias_shift: constant added to the residual (e.g. -log 4 undoes the
        #       EPAD 2.0 / PAD 0.5 loss weighting).
        #   fusion_u_only = (edges, means): drop the calibration term (a = 1) and subtract
        #       g(s) = means[bin of s] from the decision head output, keeping only d - g(s).
        self.fusion_gate_frames: int = int(getattr(lm_model, "backchannel_fusion_gate_frames", 0) or 0)
        self.fusion_bias_shift: float = 0.0
        self.fusion_u_only: Optional[tuple] = None
        self._frames_since_text: Optional[torch.Tensor] = None

    def _reset_bc_history(self) -> None:
        self._bc_hist_tout = None
        self._bc_hist_agent = None
        self._bc_hist_user = None
        self._frames_since_text = None

    def _inference_residual(self, lm_model, evidences, implicit, decision):
        """lm_model.fusion_residual plus the inference-only overrides above."""
        r = lm_model.fusion_residual(evidences, implicit, decision)
        if self.fusion_u_only is not None and decision is not None:
            edges, means = self.fusion_u_only
            edges, means = edges.to(implicit.device), means.to(implicit.device)
            idx = (torch.bucketize(implicit, edges, right=True) - 1).clamp(0, means.numel() - 1)
            r = r - means[idx]
            if lm_model.backchannel_fusion_calibration:
                r = r - (lm_model.backchannel_fusion_implicit_scale.float() - 1.0) * implicit
        if self.fusion_bias_shift:
            r = r + self.fusion_bias_shift
        if self.fusion_gate_frames > 0 and self._frames_since_text is not None:
            r = torch.where(self._frames_since_text < self.fusion_gate_frames, torch.zeros_like(r), r)
        return r

    def reset_streaming(self):
        # reset_streaming() only calls state.reset() (not _init_streaming_state), so the
        # rolling backchannel history — stored on self — must be cleared here too, otherwise
        # it leaks across clips (callers reset_streaming per utterance).
        super().reset_streaming()
        self._reset_bc_history()

    def _init_streaming_state(self, batch_size: int) -> _LMGenState:
        lm_model = self.lm_model
        initial = lm_model._get_initial_token()
        cache = torch.full(
            (batch_size, self.lm_model.num_codebooks, self.max_delay + 3),
            lm_model.ungenerated_token_id,
            device=lm_model.device,
            dtype=torch.long,
        )
        provided = torch.full(
            (batch_size, self.lm_model.num_codebooks, self.max_delay + 3),
            False,
            device=lm_model.device,
            dtype=torch.bool
        )

        disable = lm_model.device.type != 'cuda'
        # disable = True # DEBUG
        graphed_main = CUDAGraphed(lm_model.forward_codes, disable=disable)
        graphed_embeddings = CUDAGraphed(lm_model.forward_embeddings, disable=disable)
        graphed_depth = CUDAGraphed(self.depformer_step, disable=disable)

        self._reset_bc_history()
        return _LMGenState(cache, provided, initial, graphed_main, graphed_embeddings, graphed_depth)
    
    @torch.no_grad()
    def prepare_step_input(self,
                           input_tokens: torch.Tensor=None,
                           moshi_tokens:torch.Tensor=None,
                           text_token:torch.Tensor=None,
                           ):
        state = self._streaming_state
        if state is None:
            raise RuntimeError(
                "You should wrap those calls with a `with lm_gen.streaming(): ...`."
            )
        lm_model = self.lm_model

        # audio_tokens_per_stream = lm_model.dep_q//2
        needed_tokens = lm_model.num_codebooks - AUDIO_TOKENS_PER_STREAM - 1
        CT = state.cache.shape[2]

        ####
        # Fill Cache with provided tokens at state.offset (target) + delays

        if input_tokens is not None:
            assert input_tokens.dim() == 3, "Shape should be [B, K, T]."
            B, Ki, S = input_tokens.shape
            assert S == 1, "Only support being given steps one by one."
            assert (
                Ki == needed_tokens
            ), f"We expect {needed_tokens} tokens from the user stream, got {Ki}."

            for q_other in range(input_tokens.shape[1]):
                k = AUDIO_TOKENS_PER_STREAM + 1 + q_other
                delay = lm_model.delays[k]
                write_position = (state.offset + delay) % CT
                state.cache[:, k, write_position : write_position + 1] = input_tokens[:, q_other]
                state.provided[:, k, write_position : write_position + 1] = True

        if moshi_tokens is not None:
            assert moshi_tokens.dim() == 3, "Shape should be [B, K, T]."
            B, Ki, S = moshi_tokens.shape
            assert S == 1, "Only support being given steps one by one."
            assert (
                Ki == needed_tokens
            ), f"We expect {needed_tokens} tokens from the moshi stream, got {Ki}."

            for q_moshi in range(moshi_tokens.shape[1]):
                k = 1 + q_moshi
                delay = lm_model.delays[k]
                write_position = (state.offset + delay) % CT
                state.cache[:, k, write_position : write_position + 1] = moshi_tokens[:, q_moshi]
                state.provided[:, k, write_position : write_position + 1] = True

        if text_token is not None:
            write_position = (state.offset + lm_model.delays[0]) % CT
            state.cache[:, 0, write_position] = text_token
            state.provided[:, 0, write_position] = True

        for k, delay in enumerate(lm_model.delays):
            # Only for the very beginning, we extend the initial token for the acoustic
            # token that are delayed, and thus have no good value to take.
            if state.offset <= delay:
                state.cache[:, k, state.offset % CT] = state.initial[:, k, 0]
                state.provided[:, k, state.offset % CT] = True

        ####
        # Perform inference at state.offset - 1 (model_input); forcing with tokens at state.offset (target) when provided

        if state.offset == 0:
            # We can't report loss or force depth tranformer tokens until we're at step 2
            # And we need to initialize the delay-0 cache where it's not provided for step 2
            state.cache[:, :, 0] = state.initial[:, :, 0] # torch.where(state.provided[:, :, 0], state.cache[:, :, 0], state.initial[:, :, 0])
            state.offset += 1
            return None

        model_input_position = (state.offset-1) % CT
        target_position = state.offset % CT
        input_ = state.cache[:, :, model_input_position : model_input_position + 1]
        target_ = state.cache[:, :, target_position : target_position + 1]
        provided_ = state.provided[:, :, target_position : target_position + 1]

        if self.check:
            # Check that we are not feeding in any value that is not generated yet.
            assert not (input_ == lm_model.ungenerated_token_id).any(), (
                state.offset,
                input_,
            )
            assert (input_[:, lm_model.audio_offset :] <= lm_model.card).all(), input_
            assert (input_[:, :1] <= lm_model.text_card).all()
        return input_, provided_, target_, model_input_position, target_position

    @torch.no_grad()
    def step(self, input_tokens: torch.Tensor=None, moshi_tokens:torch.Tensor=None, text_token:torch.Tensor=None,
             return_embeddings: bool=False, return_z: bool=False) \
        -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        state = self._streaming_state
        lm_model = self.lm_model
        prepared_inputs = self.prepare_step_input(
            input_tokens, moshi_tokens, text_token,
        )
        # print("INPUT:", None if input_tokens is None else input_tokens.squeeze().cpu().tolist()) # DEBUG
        # print("MOSHI:", None if moshi_tokens is None else moshi_tokens.squeeze().cpu().tolist()) # DEBUG
        if prepared_inputs is None:
            if self.report_loss or self.return_logits or return_embeddings or return_z:
                return None, None
            return None
        input_, provided_, target_, model_input_position, target_position = prepared_inputs
        if self.check:
            # Check that we are not feeding in any value that is not generated yet.
            assert not (input_ == lm_model.ungenerated_token_id).any(), (
                state.offset,
                input_,
            )
            assert (input_[:, lm_model.audio_offset :] <= lm_model.card).all(), input_
            assert (input_[:, :1] <= lm_model.text_card).all()
        embeddings = None
        if return_embeddings:
            embeddings = self.lm_model.embed_codes(input_)
        transformer_out, text_logits = state.graphed_main(input_)
        output = self.process_transformer_output(
            transformer_out,
            text_logits,
            provided_,
            target_,
            model_input_position,
            target_position,
            input_codes=input_,
        )
        if return_embeddings:
            return output, embeddings
        if return_z:
            # Clone: transformer_out may alias a CUDA-graph static buffer that is
            # overwritten on every step, so the caller must not hold a raw reference.
            return output, transformer_out.clone()
        return output
    
    @torch.no_grad()
    def step_embeddings(self, embeddings: torch.Tensor):
        state = self._streaming_state
        lm_model = self.lm_model
        needed_input_tokens = lm_model.num_codebooks - AUDIO_TOKENS_PER_STREAM - 1
        _dummy_audio_token = lm_model._get_initial_token()
        while True:
            prepared_inputs = self.prepare_step_input(
                input_tokens=_dummy_audio_token[:, 1:1+needed_input_tokens], moshi_tokens=_dummy_audio_token[:, 1+needed_input_tokens:], text_token=self.zero_text_code,
            )
            if prepared_inputs is not None:
                break
        input_, provided_, target_, model_input_position, target_position = prepared_inputs
        transformer_out, text_logits = state.graphed_embeddings(embeddings)
        return self.process_transformer_output(
            transformer_out,
            text_logits,
            provided_,
            target_,
            model_input_position,
            target_position,
            input_codes=input_,
        )

    @torch.no_grad()
    def process_transformer_output(self, transformer_out, text_logits, provided_, target_,
                                   model_input_position, target_position, input_codes=None):
        state = self._streaming_state
        lm_model = self.lm_model

        # Match LMModel.forward exactly: add the checkpoint-trained START-vs-rest
        # residual to the EPAD logit, then perform the regular text sampling once.
        text_logits_for_sampling = text_logits.float()
        gate_fires = None

        # Compute explicit turn-taking evidence before text sampling.
        if lm_model.backchannel is not None:
            is_vapgpt = lm_model.backchannel_needs_audio_feats
            agent_af = None
            user_af = None
            if is_vapgpt and self.mimi is not None and input_codes is not None:
                # input_codes: [B, K, 1] — decode a single frame of per-speaker latents.
                # Sentinel values: initial_token_id = card (out-of-range high),
                # ungenerated = -2, zero = -1 (out-of-range low).
                # Clamp to [0, card-1] so mimi.decode_latent embedding lookup is always valid.
                max_code = lm_model.card - 1
                agent_af = self.mimi.decode_latent(
                    input_codes[:, 1:9, :].clamp(0, max_code)
                ).transpose(1, 2).to(dtype=transformer_out.dtype)  # [B, 1, 512]
                user_af = self.mimi.decode_latent(
                    input_codes[:, 9:17, :].clamp(0, max_code)
                ).transpose(1, 2).to(dtype=transformer_out.dtype)  # [B, 1, 512]

            if is_vapgpt:
                # Rolling-window history: the VapGPT layers have no KV cache, so replay the
                # same causal context seen during training by buffering past frames and
                # running the module over the whole window (clone/cat copies out of the
                # CUDA-graph static buffer that transformer_out may alias).
                W = self.bc_context_frames
                self._bc_hist_tout = (
                    transformer_out.clone() if self._bc_hist_tout is None
                    else torch.cat([self._bc_hist_tout, transformer_out], dim=1)
                )[:, -W:]
                tout_in = self._bc_hist_tout
                if agent_af is not None:
                    self._bc_hist_agent = (
                        agent_af if self._bc_hist_agent is None
                        else torch.cat([self._bc_hist_agent, agent_af], dim=1)
                    )[:, -W:]
                    self._bc_hist_user = (
                        user_af if self._bc_hist_user is None
                        else torch.cat([self._bc_hist_user, user_af], dim=1)
                    )[:, -W:]
                    agent_in, user_in = self._bc_hist_agent, self._bc_hist_user
                else:
                    agent_in, user_in = None, None
            else:
                # Non-VapGPT (MLP) module is per-frame; no history needed.
                tout_in, agent_in, user_in = transformer_out, agent_af, user_af

            bc_result = lm_model.backchannel(
                tout_in, emb_cb0=lm_model.depformer_text_emb, step=999_999,
                agent_audio_feat=agent_in, user_audio_feat=user_in,
            )
            # Current step's explicit decision = last frame of the windowed output.
            # 3-class head: 0=PAD, 1=EPAD (onset), 2=WORD (already speaking).
            bc_now = bc_result.bc_logits[:, -1]  # [B, 3]
            pred_cls = bc_now.argmax(dim=-1)

            implicit_log_odds = (
                text_logits[:, 0, 0, lm_model.end_of_text_padding_id]
                - text_logits[:, 0, 0, lm_model.text_padding_token_id]
            ).float()
            # The BC head predicts three mutually exclusive states:
            #   0 = remain silent (PAD), 1 = start speaking (EPAD/BC slot),
            #   2 = already speaking (WORD).
            #
            # Fusion needs evidence for a *new onset*, so compare START against
            # both ways in which a new onset is inappropriate.  The previous
            # START-vs-PAD difference ignored a high WORD score and could boost
            # EPAD even while the agent was already speaking.
            #
            # For softmax probabilities this is exactly
            #   log(P(START) / (P(PAD) + P(WORD))).
            # Computing it from logits via logsumexp is numerically stable and
            # avoids materialising probabilities.
            # Same evidences and residual as LMModel.forward (fusion_evidences /
            # fusion_residual), evaluated on the current frame only.
            evidences = lm_model.fusion_evidences(
                bc_result.bc_logits[:, -1],
                bc_result.vap_logits[:, -1] if bc_result.vap_logits is not None else None,
                bc_result.vad_logits[:, -1])
            explicit_log_odds = evidences["bc"]  # START vs (PAD ∪ WORD), for logging
            vap_agent_near = torch.sigmoid(evidences["vap"])
            decision_now = (bc_result.decision_logits[:, -1]
                            if bc_result.decision_logits is not None else None)
            # LMModel.forward는 fusion_trainable일 때만 residual을 더한다. fusion이
            # 학습되지 않았다면(no_fusion) 여기서도 backbone 로짓을 그대로 써야
            # 학습과 같은 조건이 된다.
            if lm_model.backchannel_fusion_trainable:
                explicit_residual = self._inference_residual(lm_model, evidences, implicit_log_odds, decision_now)
                # text_logits가 CUDA graph의 static buffer를 가리킬 수 있어 in-place
                # 덧셈 전에 복사한다.
                text_logits_for_sampling = text_logits.float().clone()
                text_logits_for_sampling[:, 0, 0, lm_model.end_of_text_padding_id] += (
                    explicit_residual
                )
            else:
                explicit_residual = torch.zeros_like(implicit_log_odds)
            fusion_score = implicit_log_odds + explicit_residual
            fusion_prob = torch.sigmoid(fusion_score)
            gate_fires = torch.zeros_like(pred_cls, dtype=torch.bool)

            # Expose only the current frame so external logging stays per-step.
            # gate is the ACTUAL decision — loggers read it directly.
            lm_model._last_bc_result = BackchannelOutput2(
                vap_logits=(bc_result.vap_logits[:, -1:] if bc_result.vap_logits is not None else None),
                vad_logits=bc_result.vad_logits[:, -1:],
                bc_logits=bc_result.bc_logits[:, -1:],  # [B, 1, 3] raw
                gate=gate_fires.unsqueeze(1),           # [B, 1] bool
                implicit_log_odds=implicit_log_odds.unsqueeze(1),
                explicit_log_odds=explicit_log_odds.unsqueeze(1),
                vap_agent_near=vap_agent_near.unsqueeze(1),
                fusion_score=fusion_score.unsqueeze(1),
                fusion_prob=fusion_prob.unsqueeze(1),
                decision_logits=(decision_now.unsqueeze(1) if decision_now is not None else None),
            )

        # Shape: [B, K_text=1, T=1, Card_text]. This is the only text sampling
        # call in the step, for both backbone-only and fusion modes.
        sampled_text_token = sample_token(
            text_logits_for_sampling,
            self.use_sampling,
            self.temp_text,
            self.top_k_text,
        )
        assert sampled_text_token.dim() == 3, sampled_text_token.shape
        assert sampled_text_token.shape[2] == 1
        assert sampled_text_token.shape[1] == 1, "Only one text stream supported."
        sampled_text_token = sampled_text_token[:, 0, 0]  # [B]

        if lm_model.backchannel is not None:
            lm_model._last_bc_result.gate = (
                sampled_text_token == lm_model.end_of_text_padding_id
            ).unsqueeze(1)

        next_text_token = torch.where(provided_[:, 0, 0], target_[:, 0, 0], sampled_text_token)
        if lm_model.backchannel is not None and self.fusion_gate_frames > 0:
            spoke = ((next_text_token != lm_model.text_padding_token_id)
                     & (next_text_token != lm_model.zero_token_id) & (next_text_token >= 0))
            if self._frames_since_text is None:
                self._frames_since_text = torch.full_like(next_text_token, 1_000_000)
            self._frames_since_text = torch.where(spoke, torch.zeros_like(self._frames_since_text),
                                                  self._frames_since_text + 1)

        if self.return_logits:
            sampled_audio_tokens, audio_logits = state.graphed_depth(next_text_token, transformer_out, target_[:,lm_model.audio_offset:,0], provided_[:,lm_model.audio_offset:,0]) # [B, K_audio, Card_audio]
        else:
            sampled_audio_tokens = state.graphed_depth(next_text_token, transformer_out, target_[:,lm_model.audio_offset:,0], provided_[:,lm_model.audio_offset:,0])

        state.provided[:, :, model_input_position] = False
        ####
        # Fill cache with generated tokens at state.offset (where not provided)

        state.cache[:, 0, target_position] = torch.where(
            ~state.provided[:, 0, target_position],
            sampled_text_token,
            state.cache[:, 0, target_position],
        )
        state.cache[:, 1 : lm_model.dep_q + 1, target_position] = torch.where(
            ~state.provided[:, 1 : lm_model.dep_q + 1, target_position],
            sampled_audio_tokens,
            state.cache[:, 1 : lm_model.dep_q + 1, target_position],
        )

        ####
        # Calculate loss of model logits (based on state.offset - 1) compared to target (state.offset)

        report = {}
        if self.report_loss:
            report = create_loss_report(
                state_cache=state.cache,
                lm_model=lm_model,
                text_logits=text_logits,
                audio_logits=audio_logits,
                target=target_,
                sampled_text_token=sampled_text_token,
                sampled_audio_tokens=sampled_audio_tokens,
                target_position=target_position,
            )

        ####
        # Collect outputs for state.offset - max_delay

        if state.offset <= self.max_delay:
            state.offset += 1
            if self.report_loss:
                return None, report
            if self.return_logits:
                return None, None
            else:
                return None
        
        B = state.cache.shape[0]
        CT = state.cache.shape[2]
        gen_delays_cuda = self.delays_cuda[: lm_model.dep_q + 1]
        index = (
            ((state.offset - self.max_delay + gen_delays_cuda) % CT)
            .view(1, -1, 1)
            .expand(B, -1, 1)
        )
        out = state.cache.gather(dim=2, index=index)

        state.offset += 1
        if self.report_loss:
            return out, report
        elif self.return_logits and not self.report_loss:
            return out, (text_logits.clone(), audio_logits.clone())
        else:
            return out

    def load_voice_prompt(self, voice_prompt: str):
        self.voice_prompt = voice_prompt
        raw_audio = load_audio(
            voice_prompt, self._sample_rate,
        )  # shape: (1, T) for mono

        # Normalize to -24 LUFS (mono-safe)
        raw_audio = normalize_audio(raw_audio, self._sample_rate, -24.0)

        # Keep shape (1, T) because your encoder expects channels-first
        if raw_audio.ndim == 1:
            raw_audio = raw_audio[None, :]

        self.voice_prompt_audio = raw_audio
        self.voice_prompt_cache: Optional[torch.Tensor] = None
        self.voice_prompt_embeddings: Optional[torch.Tensor] = None

    def load_voice_prompt_embeddings(self, path: str):
        self.voice_prompt = path
        # Cached embeddings may have been saved from a CUDA session; load to CPU
        # first (works everywhere), then move to the model's device below.
        state = torch.load(path, map_location="cpu")

        self.voice_prompt_audio = None
        self.voice_prompt_embeddings = state["embeddings"].to(self.lm_model.device)
        self.voice_prompt_cache = state["cache"].to(self.lm_model.device)

    def _encode_zero_frame(self) -> torch.Tensor:
        return torch.as_tensor(
            SILENCE_TOKENS,
            dtype=torch.long,
            device=self.lm_model.device,
        ).view(1, 8, 1)

    def _encode_sine_frame(self) -> torch.Tensor:
        return torch.as_tensor(
            SINE_TOKENS,
            dtype=torch.long,
            device=self.lm_model.device,
        ).view(1, 8, 1)

    def _encode_voice_prompt_frames(self, mimi):
        return encode_from_sphn(
            mimi,
            _iterate_audio(
                self.voice_prompt_audio,
                sample_interval_size=self._frame_size,
                pad=True,
            ),
            max_batch=1,
        )

    def _step_voice_prompt_frame(self,
                                 voice_prompt_frame_tokens: torch.Tensor,
                                 saved_embeddings: Optional[list[torch.Tensor]]=None,
                                 ):
        # Always use zero_text_code during voice prompt
        out = self.step(
            moshi_tokens=voice_prompt_frame_tokens,
            text_token=self.zero_text_code,
            input_tokens=self._encode_sine_frame(),
            return_embeddings=self.save_voice_prompt_embeddings,
        )
        if out is not None and self.save_voice_prompt_embeddings:
            _, embeddings = out
            saved_embeddings.append(embeddings)

    def _step_voice_prompt_core(self, mimi) -> Iterator[None]:
        """Shared core for stepping through the voice prompt.

        This generator yields at each *checkpoint* where the async wrapper may want to
        consult `is_alive`. The core itself is intentionally unaware of connection state.
        """
        if self.voice_prompt_embeddings is not None:
            # Replay stored voice prompt embeddings
            for next_embed in self.voice_prompt_embeddings:
                yield
                self.step_embeddings(next_embed)

            state = self._streaming_state
            state.cache.copy_(self.voice_prompt_cache)
            return

        elif self.voice_prompt_audio is not None:
            saved_embeddings = []
            for voice_prompt_frame_tokens in self._encode_voice_prompt_frames(mimi):
                yield
                self._step_voice_prompt_frame(
                    voice_prompt_frame_tokens,
                    saved_embeddings
                )
            # One last checkpoint before any optional save (nice-to-have for async disconnect)
            yield

            if self.save_voice_prompt_embeddings:
                # Offset int(self._streaming_state.offset) is not needed since calling step() for len(voice_prompt_frame_tokens)
                # and calling step_embeddings() for len(voice_prompt_embeddings) will increment offset by the same amount
                torch.save(
                    {
                        "embeddings": torch.stack(saved_embeddings, dim=0).detach().cpu(),
                        "cache": self._streaming_state.cache
                    },
                    splitext(self.voice_prompt)[0] + ".pt",
                )
        print('Done loading voice prompt.')

    def _step_voice_prompt(self, mimi):
        # Sync path intentionally does not support `is_alive` / disconnect checks.
        for _ in self._step_voice_prompt_core(mimi):
            pass

    async def _step_voice_prompt_async(self, mimi, is_alive: Optional[Callable]=None):
        for _ in self._step_voice_prompt_core(mimi):
            if is_alive is not None and not await is_alive():
                break

    def _step_audio_silence_core(self) -> Iterator[None]:
        # For slots of silence (default 0.5s) after voice/text prompts
        # (agent text, user audio, agent audio) : (PADs, silence, sine)
        for _ in range(self.audio_silence_frame_cnt):
            yield
            self.step(
                moshi_tokens=self._encode_zero_frame(),
                text_token=self.zero_text_code,
                input_tokens=self._encode_sine_frame(),
            )
        print('Done loading audio silence.')

    def _step_audio_silence(self):
        # Sync path intentionally does not support `is_alive` / disconnect checks.
        for _ in self._step_audio_silence_core():
            pass

    async def _step_audio_silence_async(self, is_alive: Optional[Callable]=None):
        for _ in self._step_audio_silence_core():
            if is_alive is not None and not await is_alive():
                break

    def _step_text_prompt_core(self) -> Iterator[None]:
        for text_prompt_token in self.text_prompt_tokens or ():
            yield
            self.step(
                moshi_tokens=self._encode_zero_frame(),
                text_token=text_prompt_token,
                input_tokens=self._encode_sine_frame(),
            )
        print('Done loading text prompt.')


    def _step_text_prompt(self):
        # Sync path intentionally does not support `is_alive` / disconnect checks.
        for _ in self._step_text_prompt_core():
            pass

    async def _step_text_prompt_async(self, is_alive: Optional[Callable]=None):
        for _ in self._step_text_prompt_core():
            if is_alive is not None and not await is_alive():
                break

    async def step_system_prompts_async(self, mimi, is_alive: Optional[Callable]=None):
        await self._step_voice_prompt_async(mimi, is_alive)
        await self._step_audio_silence_async(is_alive)
        await self._step_text_prompt_async(is_alive)
        await self._step_audio_silence_async(is_alive)

    def step_system_prompts(self, mimi):
        self._step_voice_prompt(mimi)
        self._step_audio_silence()
        self._step_text_prompt()
        self._step_audio_silence()

    def depformer_step(
        self,
        text_token: torch.Tensor,
        transformer_out: torch.Tensor,
        audio_tokens: torch.Tensor,
        audio_provided: torch.Tensor
    ) -> torch.Tensor:
        (B,) = text_token.shape
        prev_token = text_token
        lm_model = self.lm_model
        depformer_tokens: list[torch.Tensor] = []
        depformer_logits: list[torch.Tensor] = []
        assert not lm_model.depformer.is_streaming
        with lm_model.depformer.streaming(B):
            for cb_index in range(lm_model.dep_q):
                input_ = prev_token[:, None, None]
                logits = lm_model.forward_depformer(cb_index, input_, transformer_out)
                if self.return_logits:
                    assert logits.shape == (B, 1, 1, lm_model.card), logits.shape
                    ret_logits = logits.squeeze(dim=1).squeeze(dim=1)
                    assert ret_logits.shape == (B, lm_model.card), ret_logits.shape
                    depformer_logits.append(ret_logits.float())
                next_token = sample_token(
                    logits.float(),
                    self.use_sampling,
                    self.temp,
                    self.top_k,
                )
                assert next_token.shape == (B, 1, 1)
                next_token = next_token[:, 0, 0]  # shape is B
                prev_token = torch.where(
                    audio_provided[:, cb_index],
                    audio_tokens[:, cb_index],
                    next_token,
                )
                depformer_tokens.append(next_token)

        assert len(depformer_tokens) == lm_model.dep_q, (
            len(depformer_tokens),
            lm_model.dep_q,
        )
        tokens = torch.stack(depformer_tokens, dim=1)
        assert tokens.shape == (B, lm_model.dep_q), tokens.shape
        if self.return_logits:
            all_logits = torch.stack(depformer_logits, dim=1)
            assert all_logits.shape == (B, lm_model.dep_q, lm_model.card), all_logits.shape
            return tokens, all_logits
        else:
            return tokens
