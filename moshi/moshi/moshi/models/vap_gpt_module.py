# VapGPTBackchannelModule — drop-in replacement for BackchannelModule.
#
# Architecture difference vs. BackchannelModule:
#   BackchannelModule  : z_s → simple MLP → VAP logits
#   VapGPTBackchannelModule: z_s → two linear projections (User/Agent pseudo-streams)
#                            → VapGPT's ar_channel (per-stream GPT)
#                            → VapGPT's ar (cross-stream GPTStereo)
#                            → vap_head → VAP logits
#
# The LM produces a SINGLE hidden-state stream z_s (it has no separate speaker
# channels).  We resolve the stereo requirement by projecting z_s twice:
#
#   proj_user (z_s)  → x_user   — extracts User-side contextual features
#   proj_agent(z_s)  → x_agent  — extracts Agent-side contextual features
#
# Both streams are then processed by VapGPT's GPT layers, which model
# cross-stream turn-taking dynamics.  Pretrained VapGPT checkpoint weights
# can be loaded for ar_channel, ar, and vap_head; proj_* and bc_mlp are always
# trained from scratch.
#
# Output interface is identical to BackchannelModule (BackchannelOutput).


''' paste code from vap.modules.py '''

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from typing import Dict, Optional, Tuple


def ffn_block(
    din: int,
    dff: int,
    activation: str = "GELU",
    dropout: float = 0.0,
    bias: bool = False,
) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(din, dff, bias=bias),
        getattr(nn, activation)(),
        nn.Dropout(p=dropout),
        nn.Linear(dff, din, bias=bias),
    )


class MultiHeadAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, dim: int, num_heads: int, dropout: float, bias: bool = False):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.dim = dim

        # key, query, value projections for all heads
        self.key = nn.Linear(dim, dim, bias=bias)
        self.query = nn.Linear(dim, dim, bias=bias)
        self.value = nn.Linear(dim, dim, bias=bias)

        # head re-shapers
        self.unstack_heads = Rearrange("b t (h d) -> b h t d", h=self.num_heads)
        self.stack_heads = Rearrange("b h t d -> b t (h d)")

        # regularization
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        # output projection
        self.proj = nn.Linear(dim, dim, bias=bias)
        self.scale = 1.0 / math.sqrt(dim)

    def get_scores(self, q: torch.Tensor, k: torch.Tensor):
        """
        Arguments:
            q: (B, heads, T, D)
            k: (B, heads, T, D)

        Return:
            QK:     (B, heads, T, T)
        """
        return torch.einsum("bhid,bhjd->bhij", q, k)

    @staticmethod
    def prepare_causal_mask(T, device="cpu", dtype=torch.float32):
        mask = torch.tril(torch.ones((T, T), device=device, dtype=dtype)).view(
            1, 1, T, T
        )
        mask.requires_grad_(False)
        return mask

    def mask_scores(self, qk: torch.Tensor, mask=None):
        T = qk.size(-1)
        if mask is None:
            mask = MultiHeadAttention.prepare_causal_mask(
                T, device=qk.device, dtype=qk.dtype
            )
        qk = qk.masked_fill(mask == 0, float("-inf"))
        return qk

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        # batch size, sequence length, embedding dimensionality (n_embd)
        B, T, D = Q.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.unstack_heads(self.key(K))  # (B, heads, T, D_head)
        q = self.unstack_heads(self.query(Q))  # (B, heads, T, D_head)
        v = self.unstack_heads(self.value(V))  # (B, heads, T, D_head)

        # QK
        att = self.get_scores(q, k) * self.scale  #  (B, nh, T, T)
        att = self.mask_scores(att, mask)
        att = F.softmax(att, dim=-1)

        # Softmax, dropout, values
        y = self.attn_drop(att) @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        # re-assemble all head outputs side by side
        y = self.stack_heads(y)

        # output projection
        y = self.resid_drop(self.proj(y))
        return y, att


class MultiHeadAttentionAlibi(MultiHeadAttention):
    def __init__(self, dim: int, num_heads: int, dropout: float, bias: bool = False):
        super().__init__(dim, num_heads, dropout, bias)
        # self.m = torch.tensor(MultiHeadAttentionAlibi.get_slopes(num_heads))
        self.register_parameter(
            "m",
            nn.Parameter(torch.tensor(MultiHeadAttentionAlibi.get_slopes(num_heads))),
        )
        self.m.requires_grad_(False)
        self.mask = None

    @staticmethod
    def get_slopes(n):
        """
        * aLiBi slopes for heads.
        * m in Figure 3.
        * Source:
            - https://github.com/ofirpress/attention_with_linear_biases/blob/5b327adc6d131e28b40ba58906b30bb469483519/fairseq/models/transformer.py#L742

        Comments:

        In the paper, we only train models that have 2^a heads for some a. This function has
        some good properties that only occur when the input is a power of 2.
        To maintain that even closest_power_of_2 = 2**math.floor(math.log2(n))
        when the number of heads is not a power of 2, we use this workaround.
        """

        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio ** i for i in range(n)]

        # In the paper, we only train models that have 2^a heads for some a. This function has
        # some good properties that only occur when the input is a power of 2. To maintain that even
        # when the number of heads is not a power of 2, we use this workaround.
        if math.log2(n).is_integer():
            slopes = get_slopes_power_of_2(n)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(n))
            slopes = (
                get_slopes_power_of_2(closest_power_of_2)
                + MultiHeadAttentionAlibi.get_slopes(2 * closest_power_of_2)[0::2][
                    : n - closest_power_of_2
                ]
            )
        return slopes

    @staticmethod
    def get_relative_bias_matrix(n, num_heads, device="cpu", dtype=torch.float32):
        """Relative Bias matrix for aLiBi embeddings"""
        return (
            torch.arange(n, device=device, dtype=dtype)
            .view(1, 1, -1)
            .expand(1, num_heads, -1)
        )

    def get_alibi_mask(self, T: int, device="cpu", dtype=torch.float32):
        rel_bias_mat = MultiHeadAttentionAlibi.get_relative_bias_matrix(
            T, self.num_heads, device, dtype
        )
        alibi = rel_bias_mat * self.m.unsqueeze(0).unsqueeze(-1).to(device)

        # Causal mask (standard GPT pask)
        # lower triangle = 1
        # upper triangle = 0
        mask = MultiHeadAttention.prepare_causal_mask(T, device, dtype)  # (1, 1, T, T)
        # Repeat to get a mask for each head
        mask = mask.repeat(1, self.num_heads, 1, 1)  # (1, num_heads, T, T)
        # fill "future" information with negative infinity
        mask.masked_fill_(mask == 0, float("-inf"))

        # Add causality mask to alibi  (1, num_heads, T, T)
        alibi = alibi.unsqueeze(-2) + mask
        # alibi.requires_grad_(False)  # this should not be trained
        alibi = alibi.detach()
        return alibi

    def mask_scores(self, qk: torch.Tensor, mask=None):
        T = qk.size(-1)
        if mask is None:
            if self.mask is None or self.mask.shape[-1] < T:
                mask = self.get_alibi_mask(T, device=qk.device, dtype=qk.dtype)
                self.mask = mask
            else:
                mask = self.mask[..., :T, :T]

        # add aLiBi-mask to qk (see Figure 3.)
        # Addition/translation does not effect softmax (over each row)
        # mentioned in the original representation
        qk = qk + mask.to(qk.device)
        return qk


class TransformerLayer(nn.Module):
    """
    Transformer Layer

    Using pre-layer-normalization: https://arxiv.org/pdf/2002.04745.pdf
    Inspiration: https://nn.labml.ai/transformers/models.html
    AliBI Attention: https://ofir.io/train_short_test_long.pdf
    """

    def __init__(
        self,
        dim: int = 256,
        ffn_dim: int = 768,
        num_heads: int = 4,
        ffn_activation: str = "GELU",
        dropout: float = 0.1,
        cross_attention: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.dropout_p = dropout
        self.cross_attention = cross_attention

        self.dropout = nn.Dropout(p=dropout)
        self.ln_self_attn = nn.LayerNorm(dim)
        self.ln_ffnetwork = nn.LayerNorm(dim)
        self.mha = MultiHeadAttentionAlibi(
            dim=dim, num_heads=num_heads, dropout=dropout
        )
        self.ffnetwork = ffn_block(
            dim, ffn_dim, activation=ffn_activation, dropout=dropout
        )

        if cross_attention:
            self.ln_src_attn = nn.LayerNorm(dim)
            self.mha_cross = MultiHeadAttentionAlibi(
                dim=dim, num_heads=num_heads, dropout=dropout
            )

    def forward(
        self,
        x: torch.Tensor,
        src: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Using pre-layer-normalization: https://arxiv.org/pdf/2002.04745.pdf
        """

        # Self-attention
        z = self.ln_self_attn(x)
        self_attn, self_attn_weights = self.mha(Q=z, K=z, V=z, mask=mask)

        # Residual
        x = x + self.dropout(self_attn)

        # Cross-attention
        cross_attn_weights = None
        if self.cross_attention and src is not None:
            z = self.ln_src_attn(x)
            # https://nn.labml.ai/transformers/models.html#section-16
            # Don't normalize src... why?
            cross_attn, cross_attn_weights = self.mha_cross(
                Q=z, K=src, V=src, mask=mask
            )
            x = x + self.dropout(cross_attn)

        x = x + self.dropout(self.ffnetwork(self.ln_ffnetwork(x)))
        return x, self_attn_weights, cross_attn_weights


class TransformerStereoLayer(TransformerLayer):
    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        # sa1w: self-attention-weights 1
        # ca1w: cross-attention-weights 1
        z1, sa1w, ca1w = super().forward(x=x1, src=x2, mask=mask)
        z2, sa2w, ca2w = super().forward(x=x2, src=x1, mask=mask)
        return z1, z2, [sa1w, ca1w, sa2w, ca2w]


class GPT(nn.Module):
    """
    GPT like transformer Decoder-only class.

    * Uses AliBi attention (no positional embeddings or max-sequence-length)
    """

    def __init__(
        self,
        dim: int,
        dff_k: int = 3,
        num_layers: int = 4,
        num_heads: int = 4,
        activation: str = "GELU",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.dff = int(dim * dff_k)
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.activation = activation
        self.dropout = dropout

        self._build_layers()
        self.apply(self._init_weights)

    def _build_layers(self):
        layers = []
        for _ in range(self.num_layers):
            layers.append(
                TransformerLayer(
                    dim=self.dim,
                    ffn_dim=self.dff,
                    num_heads=self.num_heads,
                    ffn_activation=self.activation,
                    dropout=self.dropout,
                )
            )
        self.layers = nn.ModuleList(layers)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(
        self, x: torch.Tensor, attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        all_attention = []

        for layer in self.layers:
            x, self_attn_weights, _ = layer(x)
            if attention:
                all_attention.append(self_attn_weights)

        ret = {"x": x}

        if attention:
            self_attn_weights = torch.stack(all_attention, dim=1)
            ret["attn"] = self_attn_weights

        return ret


class GPTStereo(GPT):
    def _build_layers(self):
        layers = []
        for _ in range(self.num_layers):
            layers.append(
                TransformerStereoLayer(
                    dim=self.dim,
                    ffn_dim=self.dff,
                    num_heads=self.num_heads,
                    ffn_activation=self.activation,
                    dropout=self.dropout,
                    cross_attention=True,
                )
            )
        self.layers = nn.ModuleList(layers)

        # Combine output from both 'towers'
        self.combinator = Combinator(dim=self.dim, activation="GELU")

    def forward(
        self, x1: torch.Tensor, x2: torch.Tensor, attention: bool = False
    ) -> Dict[str, torch.Tensor]:

        self_attn_a = []
        self_attn_b = []
        cross_attn_a = []
        cross_attn_b = []
        for layer in self.layers:
            x1, x2, attn_list = layer(x1=x1, x2=x2)
            if attention:
                # [sa1w, ca1w, sa2w, ca2w] = attn_list
                self_attn_a.append(attn_list[0])
                cross_attn_a.append(attn_list[1])
                self_attn_b.append(attn_list[2])
                cross_attn_b.append(attn_list[3])

        x = self.combinator(x1, x2)
        ret = {"x": x, "x1": x1, "x2": x2}

        if attention:
            # B, num_layers, num_heads, N, N
            self_attn_a = torch.stack(self_attn_a, dim=1)  # stack on layer dim
            self_attn_b = torch.stack(self_attn_b, dim=1)  # stack on layer dim
            cross_attn_a = torch.stack(cross_attn_a, dim=1)  # stack on layer dim
            cross_attn_b = torch.stack(cross_attn_b, dim=1)  # stack on layer dim
            ret["self_attn"] = torch.stack([self_attn_a, self_attn_b], dim=1)
            ret["cross_attn"] = torch.stack([cross_attn_a, cross_attn_b], dim=1)
        return ret


class Combinator(nn.Module):
    """
    Combines the "ego-centric" representations from identical 'towers'
    processing channel 0 and 1. The towers are identical (shared weights)
    and therefore channel agnostic, e.g. they don't know if they process information
    from the view of speaker A or B.

    Here we have specific layers associated with each channel to join the representations
    into a single coherent space with channel information included.
    """

    def __init__(self, dim: int, activation: str = "GELU"):
        super().__init__()
        self.dim = dim

        # Channel information
        self.h0_a = nn.Linear(dim, dim, bias=False)  # Channel 0
        self.h0_b = nn.Linear(dim, dim, bias=False)  # Channel 1
        self.ln = nn.LayerNorm(self.dim)

        # Activation
        self.activation = getattr(nn, activation)()

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Combines the hidden states from identical 'towers' which have processed
        each channel from an 'ego-centric' view. However, the towers are channel agnostic
        by default (shared weights) so in this step we process the information from channel 0, 1
        separately into a joint representation.

        The final representation will (see GPTStereo -> ProjectionModel) go into a final linear
        layer to produce logits.
        """

        # Channel specific information
        ha = self.activation(self.ln(self.h0_a(x1)))
        hb = self.activation(self.ln(self.h0_b(x2)))
        h = ha + hb  # combine estimations from both parties
        return h






######## code for backchannel #########
import sys
from dataclasses import dataclass


@dataclass
class BackchannelOutput2:
    """Output of VapGPTBackchannelModule (v2 — head-only design, no gates/embeddings).

    vap_logits: [B, T, 256] — future VAP window classification (CE, external).
    vad_logits: [B, T, 2]   — CURRENT-frame voice activity logits per stream:
                              [..., 0] = user (out["x1"]), [..., 1] = agent (out["x2"]).
                              Trained with BCE against per-frame VAD targets.
    bc_logits:  [B, T, 3]   — future text-slot class: 0=PAD, 1=EPAD (backchannel
                              onset), 2=WORD (model already speaking). CE-trained.
    """
    vap_logits: torch.Tensor
    vad_logits: torch.Tensor
    bc_logits: torch.Tensor
    # Inference only (set by LMGen): the actual gate decision for the exposed frame,
    # [B, 1] bool. May differ from bc_logits argmax when a threshold rule is active —
    # loggers should read THIS, not recompute from logits.
    gate: Optional[torch.Tensor] = None
    # Inference-only fusion diagnostics (all [B, 1] when populated by LMGen).
    implicit_log_odds: Optional[torch.Tensor] = None
    # START-vs-rest log-odds from bc_logits:
    # log(P(class 1) / (P(class 0) + P(class 2))).
    explicit_log_odds: Optional[torch.Tensor] = None
    vap_agent_near: Optional[torch.Tensor] = None
    fusion_score: Optional[torch.Tensor] = None
    fusion_prob: Optional[torch.Tensor] = None
    # [B, T] decision evidence for the EPAD fusion (only when decision_head=True).
    decision_logits: Optional[torch.Tensor] = None


def _load_vap_state_dict(path: str) -> dict:
    """Load a VapGPT state dict from a .pt or .ckpt file.

    .pt  – plain state dict, load directly.
    .ckpt – Lightning checkpoint: extract state_dict, strip 'net.' prefix,
            drop 'VAP.codebook' entries, rename
            'vap_head.projection_head' → 'vap_head'.
    """
    # VapGPT checkpoints used here contain tensors and primitive metadata only.
    # Explicit weights_only avoids unsafe pickle fallback and the PyTorch warning.
    raw = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(raw, dict) and "state_dict" in raw:
        sd = raw["state_dict"]
        cleaned = {}
        for k, v in sd.items():
            if "VAP.codebook" in k:
                continue
            if "vap_head" in k:
                k = k.replace("vap_head.projection_head", "vap_head")
            cleaned[k.replace("net.", "")] = v
        return cleaned
    return raw  # plain .pt


class VapGPTBackchannelModule(nn.Module):
    """Backchannel prediction using VapGPT's cross-stream GPT architecture.

    Replaces BackchannelModule inside LMModel.  Input is the LM's single hidden-
    state stream z_s; the CPC audio encoder is NOT used.

    Two linear projections split z_s into pseudo User / Agent streams, which are
    then processed by VapGPT's per-stream GPT (ar_channel) and cross-stream
    GPTStereo (ar) to model turn-taking dynamics.

    Args:
        lm_dim: Dimension of z_s (LM hidden states).
        depformer_dim: DepthTransformer embedding dimension (for BC gate output).
        card: Audio codebook vocabulary size.
        vap_repo_path: Path to the VoiceActivityProjection repository root.
        checkpoint_path: Optional pretrained VapGPT checkpoint (.pt or .ckpt).
            ar_channel, ar, and vap_head weights are loaded from it; proj_* and
            bc_mlp are always random-initialised.
        vap_dim: Hidden dimension of VapGPT's GPT layers (default 256).
        channel_layers: Number of per-stream GPT layers (ar_channel).
        cross_layers: Number of cross-stream GPTStereo layers (ar).
        num_heads: Attention heads in GPT layers.
        dropout: Dropout probability.
        bc_hidden: Hidden size of the BC gate MLP.
        pad_token_id / epad_token_id: Silence / backchannel token IDs.
        gumbel_temp_init / gumbel_temp_min / gumbel_anneal_rate:
            DEPRECATED, ignored. The gate is now a deterministic straight-through
            argmax (no Gumbel, no temperature). Kept only for config/API compat.
    """

    def __init__(
        self,
        lm_dim: int,
        depformer_dim: int,
        card: int,
        vap_repo_path: str,
        checkpoint_path: Optional[str] = None,
        vap_dim: int = 256,
        channel_layers: int = 1,
        cross_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
        bc_hidden: int = 512,
        pad_token_id: Optional[int] = None,
        epad_token_id: Optional[int] = None,
        gumbel_temp_init: float = 1.0,
        gumbel_temp_min: float = 0.5,
        gumbel_anneal_rate: float = 0.0001,
        use_silence_ctx_proj: bool = True,
        decision_head: bool = False,
        decision_hidden: int = 128,
        aux_to_trunk: str = "full",
    ):
        super().__init__()
        if aux_to_trunk not in ("full", "detach"):
            raise ValueError(f"aux_to_trunk must be 'full' or 'detach', got {aux_to_trunk!r}")
        # aux_to_trunk="detach": the VAP / VAD / BC heads still learn from their own
        # labels, but their gradients stop at the head inputs, so the shared trunk
        # (proj_audio, ar_channel, ar, z_ctx_proj) is shaped only by the EPAD
        # objective that reaches it through the decision head.
        self.aux_to_trunk = aux_to_trunk

        self.lm_dim = lm_dim
        self.depformer_dim = depformer_dim
        self.card = card
        self.vap_dim = vap_dim
        self.pad_token_id = pad_token_id if pad_token_id is not None else card - 2
        self.epad_token_id = epad_token_id if epad_token_id is not None else card - 1
        assert self.pad_token_id < card
        assert self.epad_token_id < card

        # gumbel_temp_* are deprecated no-ops (gate is deterministic straight-through now).
        self.use_silence_ctx_proj = use_silence_ctx_proj

        # ── Pseudo-speaker projections (LM-dim fallbacks) ────────────────
        # z_s carries fused context for both speakers.  Two independent linear
        # projections let the model learn to extract user-side vs. agent-side
        # features from the same hidden state.
        # self.proj_user = nn.Linear(lm_dim, vap_dim, bias=False)   # User stream
        # self.proj_agent = nn.Linear(lm_dim, vap_dim, bias=False)  # Agent stream

        # ── Audio projections for real per-speaker Mimi latents (512-dim) ─
        # self.proj_user_audio  = nn.Linear(512, vap_dim, bias=False)
        # self.proj_agent_audio = nn.Linear(512, vap_dim, bias=False)
        self.proj_audio = nn.Linear(512, vap_dim, bias=False)

        # ── LM-context projection (z_s stream restoration, head-level late fusion) ─
        # z_s (lm_dim=4096) is projected down to vap_dim and LayerNorm'd. It is added
        # to the VAP and BC heads as ZERO-INIT additive residuals (adapter pattern):
        #   • pretrained VapGPT layers (ar_channel/ar/vap_head) keep their original
        #     shapes and inputs → checkpoint loading is untouched, and at init the
        #     module behaves exactly like the pretrained audio-only model;
        #   • vap_loss and bc_loss now backprop through z_ctx_proj into the LM
        #     backbone → multi-task shaping of the backbone representation.
        # VAD stays purely acoustic by design (current-frame VA is solvable from
        # audio alone; z_s residual excluded there).
        self.z_ctx_proj = nn.Linear(lm_dim, vap_dim, bias=False)
        self.z_ctx_norm = nn.LayerNorm(vap_dim)

        # ── VapGPT GPT layers ─────────────────────────────────────────────
        if vap_repo_path not in sys.path:
            sys.path.insert(0, vap_repo_path)

        # Per-stream autoregressive GPT (shared weights, applied to each stream)
        self.ar_channel = GPT(
            dim=vap_dim,
            dff_k=3,
            num_layers=channel_layers,
            num_heads=num_heads,
            dropout=dropout,
        )
        # Cross-stream turn-taking GPT
        self.ar = GPTStereo(
            dim=vap_dim,
            dff_k=3,
            num_layers=cross_layers,
            num_heads=num_heads,
            dropout=dropout,
        )

        # ── VAP head ──────────────────────────────────────────────────────
        # 256 = 2^(2 * n_bins) with n_bins=4 (VapGPT default).
        # NOTE: input stays vap_dim (NOT concat) so the pretrained vap_head weights
        # load cleanly; the z_s contribution enters via the zero-init residual below.
        self.vap_head = nn.Linear(vap_dim, 256)
        self.vap_z_head = nn.Linear(vap_dim, 256, bias=False)
        nn.init.zeros_(self.vap_z_head.weight)

        self.va_classifier = nn.Linear(vap_dim, 1)  # purely acoustic (no z_s)

        # BC head: audio/turn-taking pathway + zero-init z_s residual (additive,
        # symmetric with vap_head; replaces the earlier concat design).
        self.bc_head = nn.Linear(vap_dim, 3)  # BC logits (PAD/EPAD/WORD)
        self.bc_z_head = nn.Linear(vap_dim, 3, bias=False)
        nn.init.zeros_(self.bc_z_head.weight)

        # Decision head: evidence trained only through the fused EPAD text loss
        # (no auxiliary label). Reads the cross-speaker audio state and the LM
        # context. Last layer zero-initialised → the module starts exactly as the
        # model without it, while the head's own weights still get gradients.
        self.decision_head = None
        if decision_head:
            self.decision_head = nn.Sequential(
                nn.Linear(2 * vap_dim, decision_hidden), nn.GELU(), nn.Linear(decision_hidden, 1))
            nn.init.zeros_(self.decision_head[2].weight)
            nn.init.zeros_(self.decision_head[2].bias)

        # ── Load pretrained VapGPT weights (GPT layers + vap_head) ───────
        if checkpoint_path is not None:
            self._load_pretrained(checkpoint_path)

    def _load_pretrained(self, checkpoint_path: str) -> None:
        """Load ar_channel, ar, and vap_head weights from a pretrained checkpoint."""
        print(f"[VapGPTBackchannelModule] Loading pretrained checkpoint: {checkpoint_path}")
        full_sd = _load_vap_state_dict(checkpoint_path)

        # Filter to only the GPT and VAP-head keys (skip encoder.*)
        own_sd = self.state_dict()
        to_load = {}
        for k, v in full_sd.items():
            if k.startswith("ar_channel.") or k.startswith("ar.") or k.startswith("vap_head."):
                if k in own_sd and own_sd[k].shape == v.shape:
                    to_load[k] = v

        missing_in_ckpt = [k for k in own_sd if k.startswith(("ar_channel.", "ar.", "vap_head.")) and k not in to_load]
        print(f"[VapGPTBackchannelModule] Loaded {len(to_load)} GPT/vap_head tensors from checkpoint.")
        if missing_in_ckpt:
            print(f"[VapGPTBackchannelModule] {len(missing_in_ckpt)} keys not found in checkpoint (random init).")

        own_sd.update(to_load)
        self.load_state_dict(own_sd)

    def forward(
        self,
        z_s: torch.Tensor,
        emb_cb0: nn.Module,
        step: int = 0,
        agent_audio_feat: Optional[torch.Tensor] = None,
        user_audio_feat: Optional[torch.Tensor] = None,
    ) -> BackchannelOutput2:
        """Backchannel prediction from LM hidden states (v2 — pure prediction heads).

        Args:
            z_s:              [B, T, lm_dim] LM backbone hidden states.
            emb_cb0:          unused (kept for API compat with the v1 module).
            step:             unused (kept for API compat; temperature/gumbel removed).
            agent_audio_feat: [B, T, 512] real per-speaker Mimi latents for agent (optional).
            user_audio_feat:  [B, T, 512] real per-speaker Mimi latents for user (optional).

        Returns:
            BackchannelOutput2 — vap_logits [B,T,256], vad_logits [B,T,2] (current VA:
            user, agent), bc_logits [B,T,3] (PAD/EPAD/WORD).
        """
        B, T, _ = z_s.shape
        device = z_s.device

        # v2: both pseudo-speaker streams come from REAL per-speaker Mimi latents
        # (512-dim) through a single shared projection — symmetric VapGPT-style
        # input. z_s is no longer projected into the streams (proj_agent removed).
        assert user_audio_feat is not None and agent_audio_feat is not None, (
            "VapGPTBackchannelModule(v2) requires per-speaker audio latents: "
            "pass mimi to LMModel.forward_train / LMGen so agent/user feats are auto-extracted."
        )
        x_user  = self.proj_audio(user_audio_feat)   # [B, T, vap_dim]
        x_agent = self.proj_audio(agent_audio_feat)  # [B, T, vap_dim]

        h_user = self.ar_channel(x_user)["x"]   # [B, T, vap_dim]
        h_agent = self.ar_channel(x_agent)["x"] # [B, T, vap_dim]

        out = self.ar(h_user, h_agent)  # {"x", "x1", "x2"}

        # z_s stream (head-level late fusion): shared projected LM context, added
        # as zero-init residuals so training starts at the pretrained audio-only
        # behaviour and vap/bc losses backprop into the LM backbone through z_s.
        z_ctx = self.z_ctx_norm(self.z_ctx_proj(z_s))               # [B, T, vap_dim]

        # Inputs of the auxiliary (label-supervised) heads; detached when the trunk
        # should be trained only by the EPAD objective.
        if self.aux_to_trunk == "detach":
            x_aux, x1_aux, x2_aux, z_aux = out["x"].detach(), out["x1"].detach(), out["x2"].detach(), z_ctx.detach()
        else:
            x_aux, x1_aux, x2_aux, z_aux = out["x"], out["x1"], out["x2"], z_ctx

        # VAD: purely acoustic per-stream heads (z_s intentionally excluded —
        # current-frame VA is solvable from audio; keeps this head untangled).
        v1 = self.va_classifier(x1_aux)
        v2 = self.va_classifier(x2_aux)
        vad_logits = torch.cat((v1, v2), dim=-1)

        vap_logits = self.vap_head(x_aux) + self.vap_z_head(z_aux)  # [B, T, 256]
        bc_logits = self.bc_head(x_aux) + self.bc_z_head(z_aux)     # [B, T, 3]

        decision_logits = None
        if self.decision_head is not None:
            decision_logits = self.decision_head(torch.cat((out["x"], z_ctx), dim=-1)).squeeze(-1)  # [B, T]

        return BackchannelOutput2(
            vap_logits=vap_logits,
            vad_logits=vad_logits,
            bc_logits=bc_logits,
            decision_logits=decision_logits,
        )
