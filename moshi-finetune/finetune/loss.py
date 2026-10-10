import math

import torch
from torch.nn import functional as F


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Compute mean of x over positions where mask is True.

    Args:
        x:    [B, T, ...] arbitrary trailing dims
        mask: [B, T] bool — True = valid frame, False = silence-padded
    """
    m = mask
    while m.dim() < x.dim():
        m = m.unsqueeze(-1)
    m = m.expand_as(x)
    return (x * m).sum() / m.sum().clamp(min=1)


@torch.no_grad()
def _codec_quant_to_sum_feat_chunked(codec, motion: torch.Tensor) -> torch.Tensor:
    """Run ARTalkCodec.quant_to_sum_feat in 100-frame chunks (codec length constraint)."""
    chunk_len = int(getattr(codec, "patch_nums", [100])[-1])
    chunks = []
    for start in range(0, motion.shape[1], chunk_len):
        chunk = motion[:, start : start + chunk_len]
        actual_len = chunk.shape[1]
        if actual_len < chunk_len:
            pad = chunk.new_zeros(chunk.shape[0], chunk_len - actual_len, chunk.shape[2])
            chunk = torch.cat([chunk, pad], dim=1)
        feat = codec.quant_to_sum_feat(chunk)
        chunks.append(feat[:, :actual_len])
    return torch.cat(chunks, dim=1)[:, : motion.shape[1]]


def compute_face_loss(
    face_outputs: dict,
    gt_face_motion: torch.Tensor,
    codec,
    face_args,
    valid_face_mask: torch.Tensor | None = None,
    return_components: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the full reference face-motion training loss.

    Replicates the loss from softvq_continuous_online_train.py::compute_loss(),
    including motion L1, prior L1, z MSE/BCE, component-wise jaw L1, velocity,
    regularisation, and gate-target MSE terms.

    Args:
        face_outputs: dict returned by CausalSoftVQContinuousTransformer.forward().
        gt_face_motion: [B, T_face, 54] ground-truth 3DMM at 25 fps.
        codec: frozen ARTalkCodec; used to compute z_target (no_grad).
        face_args: FaceGenArgs dataclass carrying per-loss weights.

    Returns:
        Scalar loss tensor.  When ``return_components`` is True, also returns
        the unweighted component losses for monitoring.
    """
    pred   = face_outputs["pred_motion"]    # [B, T, 54]
    prior  = face_outputs["prior_motion"]   # [B, T, 54]
    z_pred = face_outputs["z_pred"]         # [B, T, code_dim]
    delta  = face_outputs["delta"]          # [B, T, 54]
    residual = face_outputs["residual"]     # [B, T, 54]
    group_gate = face_outputs["group_gate"] # [B, T, 3]

    gt = gt_face_motion

    # z-space target (frozen codec, no gradient)
    z_target = _codec_quant_to_sum_feat_chunked(codec, gt)  # [B, T, code_dim]
    z_target = z_target.to(dtype=z_pred.dtype, device=z_pred.device)

    code_dim = z_pred.shape[-1]

    # valid_face_mask: [B, T] bool — None means all frames valid (full 10s clips)
    use_mask = valid_face_mask is not None
    if use_mask:
        mask = valid_face_mask  # [B, T]

    if use_mask:
        loss_motion = _masked_mean((pred - gt).abs(), mask)
        loss_prior  = _masked_mean((prior - gt).abs(), mask)
        loss_z      = _masked_mean((z_pred - z_target).pow(2), mask)
    else:
        loss_motion = F.l1_loss(pred, gt)
        loss_prior  = F.l1_loss(prior, gt)
        loss_z      = F.mse_loss(z_pred, z_target)

    z_target_bits = (z_target > 0).to(z_pred.dtype)
    if use_mask:
        bce_per_elem = F.binary_cross_entropy_with_logits(
            z_pred * math.sqrt(code_dim), z_target_bits, reduction="none"
        )
        loss_z_bce = _masked_mean(bce_per_elem, mask)
    else:
        loss_z_bce = F.binary_cross_entropy_with_logits(
            z_pred * math.sqrt(code_dim), z_target_bits
        )

    if use_mask:
        loss_jaw = _masked_mean((pred[..., 50:51] - gt[..., 50:51]).abs(), mask)
    else:
        loss_jaw = F.l1_loss(pred[..., 50:51], gt[..., 50:51])

    pred_vel = pred[:, 1:] - pred[:, :-1]
    gt_vel   = gt[:, 1:]   - gt[:, :-1]
    if use_mask:
        vel_mask = mask[:, 1:]  # velocity is T-1 frames
        loss_vel = _masked_mean((pred_vel - gt_vel).pow(2), vel_mask)
    else:
        loss_vel = F.mse_loss(pred_vel, gt_vel)

    if use_mask:
        loss_reg = _masked_mean(delta.pow(2), mask) + _masked_mean(residual.pow(2), mask)
    else:
        loss_reg = delta.pow(2).mean() + residual.pow(2).mean()

    gate_target = group_gate.new_tensor(
        [face_args.gate_target_expr, face_args.gate_target_jaw, face_args.gate_target_neck]
    ).view(1, 1, 3)
    gate_group_weight = group_gate.new_tensor(
        [face_args.gate_loss_expr_weight, face_args.gate_loss_jaw_weight, face_args.gate_loss_neck_weight]
    ).view(1, 1, 3)
    if use_mask:
        loss_gate = _masked_mean((group_gate - gate_target) ** 2 * gate_group_weight, mask)
    else:
        loss_gate = ((group_gate - gate_target) ** 2 * gate_group_weight).mean()

    total = (
        face_args.motion_weight * loss_motion
        + face_args.prior_weight * loss_prior
        + face_args.z_weight * loss_z
        + face_args.z_bce_weight * loss_z_bce
        + face_args.jaw_weight * loss_jaw
        + face_args.vel_weight * loss_vel
        + face_args.reg_weight * loss_reg
        + face_args.gate_weight * loss_gate
    )
    if return_components:
        if use_mask:
            pred_jaw_abs = _masked_mean(pred[..., 50:51].abs(), mask)
            gt_jaw_abs = _masked_mean(gt[..., 50:51].abs(), mask)
            pred_jaw_velocity_abs = _masked_mean(pred_vel[..., 50:51].abs(), vel_mask)
            gt_jaw_velocity_abs = _masked_mean(gt_vel[..., 50:51].abs(), vel_mask)
            prior_jaw_abs = _masked_mean(prior[..., 50:51].abs(), mask)
            delta_jaw_abs = _masked_mean(delta[..., 50:51].abs(), mask)
            residual_jaw_abs = _masked_mean(residual[..., 50:51].abs(), mask)
            gate_jaw = _masked_mean(group_gate[..., 1:2], mask)
        else:
            pred_jaw_abs = pred[..., 50:51].abs().mean()
            gt_jaw_abs = gt[..., 50:51].abs().mean()
            pred_jaw_velocity_abs = pred_vel[..., 50:51].abs().mean()
            gt_jaw_velocity_abs = gt_vel[..., 50:51].abs().mean()
            prior_jaw_abs = prior[..., 50:51].abs().mean()
            delta_jaw_abs = delta[..., 50:51].abs().mean()
            residual_jaw_abs = residual[..., 50:51].abs().mean()
            gate_jaw = group_gate[..., 1].mean()
        return total, {
            "motion_loss": loss_motion,
            "prior_loss": loss_prior,
            "z_mse_loss": loss_z,
            "z_bce_loss": loss_z_bce,
            "jaw_loss": loss_jaw,
            "velocity_loss": loss_vel,
            "regularization_loss": loss_reg,
            "gate_loss": loss_gate,
            "pred_jaw_abs": pred_jaw_abs,
            "gt_jaw_abs": gt_jaw_abs,
            "jaw_amplitude_ratio": pred_jaw_abs / gt_jaw_abs.clamp(min=1e-8),
            "pred_jaw_velocity_abs": pred_jaw_velocity_abs,
            "gt_jaw_velocity_abs": gt_jaw_velocity_abs,
            "prior_jaw_abs": prior_jaw_abs,
            "delta_jaw_abs": delta_jaw_abs,
            "residual_jaw_abs": residual_jaw_abs,
            "gate_jaw": gate_jaw,
        }
    return total


@torch.no_grad()
def epad_confusion_counts(
    text_logits: torch.Tensor,
    target: torch.Tensor,
    target_mask: torch.Tensor,
    epad_id: int,
) -> torch.Tensor:
    """Confusion-matrix counts for [EPAD] prediction by the text head.

    Treats "is the predicted text token [EPAD]?" as a binary classification
    against the ground-truth text token, restricted to valid (masked) frames.

    Args:
        text_logits: [B, Kt, T, V] text head logits.
        target:      [B, Kt, T]    ground-truth text tokens.
        target_mask: [B, Kt, T]    bool — True = valid frame counted in the loss.
        epad_id:     end-of-text-padding token id (model.end_of_text_padding_id).

    Returns:
        Float tensor [tp, fp, fn, tn] on the logits' device (sum over all valid
        positions). Accumulate these across microbatches/ranks, then derive
        acc/precision/recall/F1 with `epad_metrics_from_counts`.
    """
    pred = text_logits.argmax(dim=-1)          # [B, Kt, T]
    valid = target_mask.bool()
    is_pred = pred == epad_id
    is_tgt = target == epad_id
    tp = (valid & is_pred & is_tgt).sum()
    fp = (valid & is_pred & ~is_tgt).sum()
    fn = (valid & ~is_pred & is_tgt).sum()
    tn = (valid & ~is_pred & ~is_tgt).sum()
    return torch.stack([tp, fp, fn, tn]).float()


def turn_end_mask(
    user_vad: torch.Tensor,
    agent_speaking: torch.Tensor | None,
    window: int,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """[B, T] bool: frames right after the user stopped speaking while the agent is silent.

    user_vad:       [B, T] user voice-activity target (1 active, 0 silent, <0 unknown).
    agent_speaking: [B, T] bool, the agent emitted a non-PAD token recently (or None).
    window:         the user must have been active in at least one of the previous
                    `window` frames and be silent now.
    """
    known = user_vad >= 0
    active = (user_vad > 0.5) & known
    T = active.shape[-1]
    before = F.pad(active.to(torch.int32).cumsum(-1), (1, 0))[..., :T]   # active frames in 0 .. t-1
    recent = (before - F.pad(before, (window, 0))[..., :T]) > 0          # ... in t-window .. t-1
    mask = known & ~active & recent
    if agent_speaking is not None:
        mask = mask & ~agent_speaking
    if valid_mask is not None:
        mask = mask & valid_mask.bool()
    return mask


BC_CLASS_NAMES = ("pad", "epad", "word")  # index = class id in bc_logits


def _dilate(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """Dilate a [B, T] bool mask by ±radius frames along the time axis."""
    return (
        F.max_pool1d(
            mask.float().unsqueeze(1),
            kernel_size=2 * radius + 1,
            stride=1,
            padding=radius,
        )
        .squeeze(1)
        .bool()
    )


def build_bc_targets(
    mode: str,
    text_tokens: torch.Tensor,
    pad_id: int,
    epad_id: int,
    onset_ignore_frames: int = 2,
    valid_mask: torch.Tensor | None = None,
    vap_targets: torch.Tensor | None = None,
    vad_targets: torch.Tensor | None = None,
    vap_horizon_bins: int = 1,
    vap_require_user_silent: bool = True,
) -> torch.Tensor:
    """Build the 3-class supervision target for the backchannel head.

    Both modes produce the same label space, so `bc_logits [B, T, 3]` and the
    Logit-Adjustment CE downstream are identical — only the definition of what
    counts as a backchannel changes.

        0 = PAD   — nothing to do
        1 = EPAD  — backchannel slot (the positive class)
        2 = WORD  — the agent is already speaking
        -100      — ignored by the loss

    Modes:
        "epad": read straight off the ground-truth text stream. EPAD marks the
            backchannels the dataset agent actually produced. PAD frames within
            ±`onset_ignore_frames` of a true EPAD are ignored, since onset labels
            carry ±1–2 frame jitter and punishing near-misses teaches hedging.

        "vap": derive the label from turn-taking dynamics in the VAP manifest.
            WORD  = agent voiced in the CURRENT frame (manifest VAD, agent row;
                    falls back to "GT text token is a word" where VAD is unknown).
            EPAD  = agent silent now but starting to speak within the next
                    `vap_horizon_bins` VAP bins — and, when
                    `vap_require_user_silent`, the user stays silent over that
                    same span.
            PAD   = everything else with a known VAP label.
            Frames with no VAP label are ignored. The same ±K boundary-ignore
            band is applied around positives.

            This labels *opportunities* rather than realised responses, so the
            positive class is denser than in "epad" mode and does not depend on
            the dataset agent having actually reacted.

    Args:
        mode:                "epad" or "vap".
        text_tokens:         [B, T] ground-truth text tokens (undelayed).
        pad_id / epad_id:    model.text_padding_token_id / .end_of_text_padding_id.
        onset_ignore_frames: ±K boundary-ignore radius. 0 disables.
        valid_mask:          [B, T] bool — False frames are ignored (silence padding).
        vap_targets:         [B, T] long VAP labels, -100 = unknown ("vap" mode only).
        vad_targets:         [B, 2, T] float current-frame VA (row0=user, row1=agent),
                             -100 = unknown ("vap" mode only, optional).
        vap_horizon_bins:    1–4 future VAP bins treated as "speaking soon".
        vap_require_user_silent: require the user to be silent over the horizon.

    Returns:
        [B, T] long tensor of class ids with -100 for ignored frames.
    """
    device = text_tokens.device
    txt = text_tokens
    is_epad_tok = txt == epad_id
    is_pad_tok = txt == pad_id
    # Real word: not PAD/EPAD and not BOS(1)/EOS(2).
    is_word_tok = (~is_pad_tok) & (~is_epad_tok) & (txt != 1) & (txt != 2) & (txt >= 0)
    K = int(onset_ignore_frames)

    if mode == "epad":
        cls_tgt = torch.full_like(txt, -100)
        cls_tgt[is_pad_tok] = 0
        cls_tgt[is_word_tok] = 2
        cls_tgt[is_epad_tok] = 1
        positives = is_epad_tok

    elif mode == "vap":
        if vap_targets is None:
            raise ValueError(
                "bc_target_mode='vap' requires VAP targets, but batch.vap_targets "
                "is None. Set data.vap_manifest to a manifest that covers this "
                "dataset, or switch backchannel.bc_target_mode back to 'epad'."
            )
        lbl = vap_targets.to(device).long()          # [B, T], -100 = unknown
        known = lbl != -100
        safe = lbl.clamp(min=0)

        # Label layout after the interleaver's speaker normalisation:
        #   bits 7..4 = user  (bit 7-i = bin i),  bits 3..0 = agent (bit 3-i = bin i)
        H = max(1, min(4, int(vap_horizon_bins)))
        agent_soon = torch.zeros_like(known)
        user_soon = torch.zeros_like(known)
        for i in range(H):
            agent_soon |= ((safe >> (3 - i)) & 1).bool()
            user_soon |= ((safe >> (7 - i)) & 1).bool()

        # Current-frame agent activity: manifest VAD where available, else the
        # text stream (a real word token means the agent is speaking).
        if vad_targets is not None:
            agent_va = vad_targets.to(device)[:, 1]  # [B, T], -100 = unknown
            agent_now = torch.where(agent_va >= 0, agent_va > 0.5, is_word_tok)
        else:
            agent_now = is_word_tok

        positives = known & (~agent_now) & agent_soon
        if vap_require_user_silent:
            positives = positives & (~user_soon)

        cls_tgt = torch.full_like(lbl, -100)
        cls_tgt[known] = 0
        cls_tgt[known & agent_now] = 2
        cls_tgt[positives] = 1

    else:
        raise ValueError(f"unknown bc_target_mode: {mode!r} (expected 'epad' or 'vap')")

    # Boundary-ignore band: PAD frames adjacent to a positive are ambiguous.
    if K > 0 and positives.any():
        band = _dilate(positives, K)
        cls_tgt[band & (cls_tgt == 0)] = -100

    if valid_mask is not None:
        cls_tgt = cls_tgt.masked_fill(~valid_mask.to(device), -100)

    return cls_tgt


def bc_head_confusion_counts(
    bc_logits: torch.Tensor, cls_tgt: torch.Tensor
) -> torch.Tensor:
    """Confusion counts for the BC head's positive class (1 = backchannel slot).

    Unlike `epad_confusion_counts` — which scores the TEXT head against the
    [EPAD] token and is therefore identical across bc_target_mode settings —
    this scores `bc_logits` against whatever target build_bc_targets produced.
    Use it to compare how learnable the "epad" and "vap" targets are; the two
    are NOT comparable to each other in absolute terms, since they define
    different positives.

    Args:
        bc_logits: [B, T, 3] backchannel head logits (prompt prefix stripped).
        cls_tgt:   [B, T] class ids from build_bc_targets, -100 = ignored.

    Returns:
        Float tensor [tp, fp, fn, tn] on the logits' device.
    """
    valid = cls_tgt != -100
    pred_bc = bc_logits.argmax(dim=-1) == 1
    tgt_bc = cls_tgt == 1
    tp = (valid & pred_bc & tgt_bc).sum()
    fp = (valid & pred_bc & ~tgt_bc).sum()
    fn = (valid & ~pred_bc & tgt_bc).sum()
    tn = (valid & ~pred_bc & ~tgt_bc).sum()
    return torch.stack([tp, fp, fn, tn]).float()


def epad_metrics_from_counts(
    counts: torch.Tensor, prefix: str = "epad"
) -> dict[str, float]:
    """Derive accuracy/precision/recall/F1 from [tp, fp, fn, tn] counts."""
    tp, fp, fn, tn = (float(x) for x in counts.tolist())
    total = tp + fp + fn + tn
    acc = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {
        f"{prefix}_acc": acc,
        f"{prefix}_precision": precision,
        f"{prefix}_recall": recall,
        f"{prefix}_f1": f1,
    }


def compute_loss_with_mask(
    logits: torch.Tensor,
    target: torch.Tensor,
    target_mask: torch.Tensor,
    mode: str,
    first_codebook_weight_multiplier: float = 1.0,
    text_padding_weight: float = 1.0,
    text_padding_ids: set[int] | None = None,
    epad_weight: float = 1.0,
    epad_ids: set[int] | None = None,
):
    target = torch.where(target_mask, target, torch.zeros_like(target))

    weights = target_mask.float()
    if mode == "audio":
        weights[:, 0] *= first_codebook_weight_multiplier
    elif mode == "text":
        assert text_padding_ids is not None
        for id in text_padding_ids:
            weights[target == id] *= text_padding_weight
        if epad_ids is not None:
            for id in epad_ids:
                weights[target == id] *= epad_weight

    logits = logits.reshape(-1, logits.size(-1)).float()
    target = target.reshape(-1)
    weights = weights.reshape(-1)
    mb_loss = F.cross_entropy(logits, target, reduction="none")
    mb_loss = torch.where(weights > 0.0, mb_loss * weights, torch.zeros_like(mb_loss))
    mb_loss = torch.sum(mb_loss) / torch.sum(weights).clamp(min=1e-8)

    return mb_loss
