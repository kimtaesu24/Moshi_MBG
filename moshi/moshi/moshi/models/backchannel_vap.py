# Copyright (c) 2026, Backchannel VAP Module for Moshi/Helium
#
# Inserts between Helium backbone hidden states and DepthTransformer
# to conditionally replace audio tokens with backchannel signals.
#
# Design:
#   Step 2:  VAP MLP         z_s → z_vap ∈ ℝ^vap_dim
#   Step 3:  BC MLP          [z_vap, z_s] → z_bc ∈ ℝ^2
#   Step 4:  Gumbel-Softmax  z_bc → y_bc ∈ {0,1}  (EPAD decision)
#   Step 5:  Silence gate    silence_gate_mlp(z_s.detach()) → s_pad (Alt 1)
#            g_final = s_pad * y_bc
#   Step 6:  bc_embeddings used at cb_index=1 input in depformer (Alt 2)
#   Alt 3:   Commitment loss computed in lm.py from depformer cb_index=0 logits

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional


@dataclass
class BackchannelOutput:
    bc_embeddings: torch.Tensor        # [B, T, depformer_dim] — gated emb for cb_index=1 (Alt 2)
    vap_logits: torch.Tensor           # [B, T, vap_dim] — for VAP CE loss (external)
    bc_gate: torch.Tensor              # [B, T] — binary gate (1 = insert backchannel)
    bc_logits: torch.Tensor            # [B, T, 2] — raw z_bc for commitment loss (Alt 3)
    silence_gate_logits: torch.Tensor  # [B, T, 2] — silence gate logits for commitment loss (Alt 3)


def gumbel_softmax_st(
    logits: torch.Tensor,
    temperature: float = 1.0,
    hard: bool = True,
) -> torch.Tensor:
    """Gumbel-Softmax with Straight-Through estimator.

    Returns [*, 2] one-hot (hard=True) or soft probabilities.
    """
    if torch.is_grad_enabled():
        gumbels = -torch.empty_like(logits).exponential_().log()
        gumbels = (logits + gumbels) / temperature
    else:
        gumbels = logits / temperature
    y_soft = F.softmax(gumbels, dim=-1)
    if hard:
        index = y_soft.max(dim=-1, keepdim=True)[1]
        y_hard = torch.zeros_like(y_soft).scatter_(-1, index, 1.0)
        y = (y_hard - y_soft).detach() + y_soft
    else:
        y = y_soft
    return y


def compute_temperature(step: int, init: float = 1.0, min_temp: float = 0.5,
                        anneal_rate: float = 0.0001) -> float:
    """Exponential temperature annealing: max(min_temp, init * exp(-anneal_rate * step))."""
    return max(min_temp, init * math.exp(-anneal_rate * step))


class BackchannelModule(nn.Module):
    """Backchannel prediction and insertion module.

    Placed between Helium backbone output (z_s) and DepthTransformer input.
    Uses three complementary mechanisms (Alt 1/2/3) described in the module docstring above.

    Key change from original design:
        - No double DepthTransformer run (Alt 1 replaces the self-reflective peek).
        - bc_embeddings targets cb_index=0 input (텍스트 공간). PAD/EPAD는 텍스트 어휘
          토큰이므로 emb_cb0으로 depformer_text_emb를 전달해야 함.
        - Commitment loss (Alt 3) trains silence_gate_mlp and bc_mlp to agree with the
          DepthTransformer's natural PAD probability — computed externally in lm.py.
    """

    def __init__(
        self,
        lm_dim: int,
        depformer_dim: int,
        card: int,
        vap_dim: int = 256,
        bc_hidden: int = 512,
        pad_token_id: Optional[int] = None,
        epad_token_id: Optional[int] = None,
        gumbel_temp_init: float = 1.0,
        gumbel_temp_min: float = 0.5,
        gumbel_anneal_rate: float = 0.0001,
    ):
        super().__init__()
        self.lm_dim = lm_dim
        self.depformer_dim = depformer_dim
        self.card = card
        self.vap_dim = vap_dim
        self.pad_token_id = pad_token_id if pad_token_id is not None else card - 2
        self.epad_token_id = epad_token_id if epad_token_id is not None else card - 1
        assert self.pad_token_id < card, f"pad_token_id {self.pad_token_id} must be < {card}"
        assert self.epad_token_id < card, f"epad_token_id {self.epad_token_id} must be < {card}"
        self.gumbel_temp_init = gumbel_temp_init
        self.gumbel_temp_min = gumbel_temp_min
        self.gumbel_anneal_rate = gumbel_anneal_rate

        # Step 2: VAP MLP — z_s → z_vap ∈ ℝ^vap_dim
        self.vap_mlp = nn.Sequential(
            nn.Linear(lm_dim, bc_hidden),
            nn.ReLU(),
            nn.Linear(bc_hidden, vap_dim),
        )

        # Step 3: BC MLP — [z_vap, z_s] → z_bc ∈ ℝ^2
        self.bc_mlp = nn.Sequential(
            nn.Linear(vap_dim + lm_dim, bc_hidden),
            nn.ReLU(),
            nn.Linear(bc_hidden, 2),
        )

        # Alt 1: Silence gate — lightweight replacement for the self-reflective depformer peek.
        # Operates on z_s.detach() so it cannot disturb Helium's gradients.
        # Trained by commitment loss (Alt 3) to track DepthTransformer's PAD probability.
        self.silence_gate_mlp = nn.Sequential(
            nn.Linear(lm_dim, bc_hidden // 2),
            nn.ReLU(),
            nn.Linear(bc_hidden // 2, 2),
        )

    def get_temperature(self, step: int) -> float:
        return compute_temperature(
            step, self.gumbel_temp_init, self.gumbel_temp_min, self.gumbel_anneal_rate)

    def forward(
        self,
        z_s: torch.Tensor,
        emb_cb0: nn.Module,
        step: int = 0,
    ) -> BackchannelOutput:
        """Run backchannel prediction and gating.

        Args:
            z_s:     [B, T, lm_dim] Helium backbone hidden states.
            emb_cb0: PAD/EPAD 토큰을 조회할 임베딩 테이블. PAD/EPAD가 텍스트 어휘에
                     속하므로 호출 측에서 depformer_text_emb를 전달해야 함.
                     bc_embeddings는 cb_index=0 입력 교체에 사용됨 (Alt 2).
            step:    current training step for temperature annealing.

        Returns:
            BackchannelOutput — see field docs for how each tensor is used.
        """
        B, T, _ = z_s.shape
        temp = self.get_temperature(step)
        device = z_s.device

        # Step 2: VAP prediction
        z_vap = self.vap_mlp(z_s)  # [B, T, vap_dim]

        # Step 3: BC score
        z_bc = self.bc_mlp(torch.cat([z_vap, z_s], dim=-1))  # [B, T, 2]

        # Step 4: BC Gumbel-Softmax → y_bc  (index 1 = EPAD / insert backchannel)
        y_bc_onehot = gumbel_softmax_st(z_bc, temperature=temp, hard=True)
        y_bc = y_bc_onehot[..., 1]  # [B, T], hard {0,1} for g_final

        # Alt 1: Silence gate on detached z_s.
        s_pad_logits = self.silence_gate_mlp(z_s.detach())  # [B, T, 2]
        s_pad_onehot = gumbel_softmax_st(s_pad_logits, temperature=temp, hard=True)
        s_pad = s_pad_onehot[..., 1]  # [B, T], hard {0,1} for g_final

        # Hard gate used at inference for discrete token replacement.
        g_final = s_pad * y_bc  # [B, T]

        # Soft gate for training embedding — product of raw softmax probs (no Gumbel, no ST).
        # Both gates contribute gradient at every timestep; no blocking from the other being 0.
        y_bc_soft = F.softmax(z_bc, dim=-1)[..., 1]          # [B, T]
        s_pad_soft = F.softmax(s_pad_logits, dim=-1)[..., 1]  # [B, T]
        g_soft = y_bc_soft * s_pad_soft                        # [B, T], always in (0,1)

        pad_ids = torch.full((1,), self.pad_token_id, device=device, dtype=torch.long)
        epad_ids = torch.full((1,), self.epad_token_id, device=device, dtype=torch.long)
        pad_emb = emb_cb0(pad_ids)    # [1, depformer_dim]
        epad_emb = emb_cb0(epad_ids)  # [1, depformer_dim]
        g_soft_exp = g_soft.unsqueeze(-1)
        bc_token_emb = g_soft_exp * epad_emb + (1.0 - g_soft_exp) * pad_emb  # [B, T, depformer_dim]

        '''
        # hyades_taesoo version (train with)
        y_bc_exp = y_bc.unsqueeze(-1)
        bc_token_emb = y_bc_exp * epad_emb + (1.0 - y_bc_exp) * pad_emb
        '''

        return BackchannelOutput(
            bc_embeddings=bc_token_emb,
            vap_logits=z_vap,
            bc_gate=g_final,
            bc_logits=z_bc,
            silence_gate_logits=s_pad_logits,
        )
