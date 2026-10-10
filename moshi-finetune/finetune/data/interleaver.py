import json
import logging
import math
import os
import re
from collections import deque
from dataclasses import dataclass
from functools import reduce
from pathlib import Path
from typing import Optional

import numpy as np
import sentencepiece
import torch
# moshi.conditioners does not exist in the personaplex version of moshi.
# ConditionAttributes is defined here as a local stub — personaplex training
# does not use condition tensors, so this is only needed as a type placeholder.
from dataclasses import field
from typing import Any


logger = logging.getLogger(__name__)


@dataclass
class ConditionAttributes:
    """Stub replacement for moshi.conditioners.ConditionAttributes.
    Personaplex training does not use conditioning, so this is empty.
    """
    text: dict[str, Any] = field(default_factory=dict)
    wav: dict[str, Any] = field(default_factory=dict)

Alignment = tuple[str, tuple[float, float], str]
TokenizedAlignment = tuple[list[int], tuple[float, float], str]


@dataclass
class Sample:
    codes: torch.Tensor
    condition_attributes: ConditionAttributes | None = None
    vap_targets: torch.Tensor | None = None  # [1, T]
    voice_prompt_emb: torch.Tensor | None = None  # [V, dim]
    face_motion_gt: torch.Tensor | None = None    # [T_face, 54] at 25 fps
    valid_mask: torch.Tensor | None = None        # [T_mimi] bool: True = real audio, False = silence-padded
    valid_face_frames: int | None = None          # number of valid face frames (25 fps)
    bc_timing_targets: torch.Tensor | None = None  # [1, T] float32: 1=good BC, 0=not, -100=unknown
    vad_targets: torch.Tensor | None = None        # [1, 2, T] float32 current-frame VA (row0=user, row1=agent), -100=unknown


@dataclass
class Batch:
    codes: torch.Tensor
    condition_attributes: list[ConditionAttributes] | None = None
    vap_targets: torch.Tensor | None = None       # [B, T]
    voice_prompt_embs: torch.Tensor | None = None  # [B, V, dim]
    face_motion_gt: torch.Tensor | None = None    # [B, T_face, 54] at 25 fps
    valid_mask: torch.Tensor | None = None        # [B, T_mimi] bool
    valid_face_frames: torch.Tensor | None = None # [B] int: valid face frame count per sample
    bc_timing_targets: torch.Tensor | None = None  # [B, T] float32
    vad_targets: torch.Tensor | None = None        # [B, 2, T] float32 current-frame VA (user, agent), -100=unknown

    @classmethod
    def collate(cls, batch: list[Sample]) -> "Batch":
        codes = torch.cat([b.codes for b in batch])
        vap_targets = None
        if batch[0].vap_targets is not None:
            vap_targets = torch.cat([b.vap_targets for b in batch])

        voice_prompt_embs = None
        if all(b.voice_prompt_emb is not None for b in batch):
            # cache shape is fixed [1, K, T] int64 — cat along batch dim, no padding needed
            voice_prompt_embs = torch.cat([b.voice_prompt_emb for b in batch], dim=0)  # [B, K, T]

        face_motion_gt = None
        if all(b.face_motion_gt is not None for b in batch):
            # Pad to max T_face in this batch before stacking
            max_t = max(b.face_motion_gt.shape[0] for b in batch)
            padded_face = []
            for b in batch:
                fm = b.face_motion_gt  # [T_face, 54]
                if fm.shape[0] < max_t:
                    pad = torch.zeros(max_t - fm.shape[0], fm.shape[1], dtype=fm.dtype)
                    fm = torch.cat([fm, pad], dim=0)
                padded_face.append(fm)
            face_motion_gt = torch.stack(padded_face, dim=0)  # [B, T_face, 54]

        valid_mask = None
        if all(b.valid_mask is not None for b in batch):
            valid_mask = torch.stack([b.valid_mask for b in batch], dim=0)  # [B, T_mimi]

        valid_face_frames = None
        if all(b.valid_face_frames is not None for b in batch):
            valid_face_frames = torch.tensor(
                [b.valid_face_frames for b in batch], dtype=torch.long
            )  # [B]

        bc_timing_targets = None
        if all(b.bc_timing_targets is not None for b in batch):
            bc_timing_targets = torch.cat([b.bc_timing_targets for b in batch])  # [B, T]

        vad_targets = None
        if all(b.vad_targets is not None for b in batch):
            vad_targets = torch.cat([b.vad_targets for b in batch])  # [B, 2, T]

        if batch[0].condition_attributes is None:
            return Batch(codes, vap_targets=vap_targets, voice_prompt_embs=voice_prompt_embs,
                         face_motion_gt=face_motion_gt, valid_mask=valid_mask,
                         valid_face_frames=valid_face_frames,
                         bc_timing_targets=bc_timing_targets,
                         vad_targets=vad_targets)
        return Batch(codes, [b.condition_attributes for b in batch], vap_targets=vap_targets,
                     voice_prompt_embs=voice_prompt_embs, face_motion_gt=face_motion_gt,
                     valid_mask=valid_mask, valid_face_frames=valid_face_frames,
                     bc_timing_targets=bc_timing_targets,
                     vad_targets=vad_targets)


def tokenize(
    tokenizer: sentencepiece.SentencePieceProcessor,
    text: str,
    bos: bool = True,
    alpha: float | None = None,
):
    """Tokenize the given string, accounting for new lines, potentially adding a BOS token."""
    nl_piece = tokenizer.encode("\n")[-1]
    if alpha is not None:
        tokens = tokenizer.encode(
            text.split("\n"), enable_sampling=True, alpha=alpha, nbest_size=-1
        )
    else:
        tokens = tokenizer.encode(text.split("\n"))
    tokens = reduce(lambda a, b: [*a, nl_piece, *b], tokens)
    if bos:
        tokens = [tokenizer.bos_id(), *tokens]
    return tokens


class Interleaver:
    """Interleaver with basic featuress
    Args:
        tokenizer: text tokenizer used by the model.
        audio_frame_rate (float): frame rate of the audio tokenizer.
        text_padding (int): special token used for text padding.
        end_of_text_padding (int): special token used to indicate end of text padding.
        zero_padding (int): special token id indicating that a 0 should be used instead
            of an actual embedding.
        in_word_padding (int | None): padding used within a word segment. Will default to `text_padding`.
        keep_main_only (bool): if True, will only keep the alignments with the main speaker.
        keep_and_shift (bool): if True, will not drop any alignment, except for those with negative duration.
        use_bos_eos: (bool): if True, inserts BOS, EOS for change of turns.
        audio_delay (float): delay between the text and audio.
            A positive value means the text will be ahead of the audio.
        proba (float): probability of keeping the text.
        device: device location for the output tensors.
    """

    def __init__(
        self,
        tokenizer: sentencepiece.SentencePieceProcessor,
        audio_frame_rate: float,
        text_padding: int,
        end_of_text_padding: int,
        zero_padding: int,
        in_word_padding: int | None = None,
        keep_main_only: bool = False,
        main_speaker_label: str = "SPEAKER_MAIN",
        use_bos_eos: bool = False,
        keep_and_shift: bool = False,
        audio_delay: float = 0.0,
        proba: float = 1.0,
        device: str | torch.device = "cuda",
    ):
        self.tokenizer = tokenizer
        self.audio_frame_rate = audio_frame_rate
        self.text_padding = text_padding
        self.end_of_text_padding = end_of_text_padding
        self.zero_padding = zero_padding
        self.in_word_padding = (
            self.text_padding if in_word_padding is None else in_word_padding
        )
        self.keep_main_only = keep_main_only
        self.main_speaker_label = main_speaker_label
        self.use_bos_eos = use_bos_eos
        self.keep_and_shift = keep_and_shift
        self.audio_delay = audio_delay
        self.proba = proba
        self.device = device

    @property
    def special_tokens(self) -> set[int]:
        """Return the set of special tokens used by this interleaver."""
        return {
            self.text_padding,
            self.end_of_text_padding,
            self.tokenizer.bos_id(),
            self.tokenizer.eos_id(),
            self.zero_padding,
            self.in_word_padding,
        }

    def _tokenize(self, alignments: list[Alignment]) -> list[TokenizedAlignment]:
        # Tokenizes each word individually into a list of ints.
        out = []
        for word, ts, speaker in alignments:
            toks = tokenize(self.tokenizer, word.strip(), bos=False)
            out.append((toks, ts, speaker))
        return out

    def _keep_main_only(
        self, alignments: list[TokenizedAlignment], main_speaker: str
    ) -> list[TokenizedAlignment]:
        return [a for a in alignments if a[2] == main_speaker]

    def _keep_those_with_duration(
        self, alignments: list[TokenizedAlignment]
    ) -> list[TokenizedAlignment]:
        # Removes all words with negative or 0 durations.
        return [a for a in alignments if a[1][0] < a[1][1]]

    def _add_delay(
        self, alignments: list[TokenizedAlignment]
    ) -> list[TokenizedAlignment]:
        # Delay the audio with respect to the text, e.g. positive values mean the audio is late on the text.
        return [
            (a[0], (a[1][0] - self.audio_delay, a[1][1] - self.audio_delay), a[2])
            for a in alignments
            if a[1][1] > self.audio_delay
        ]

    def _insert_bos_eos(
        self, alignments: list[TokenizedAlignment], main_speaker: str
    ) -> list[TokenizedAlignment]:
        # EOS and BOS is different from what it was in the old Interleaver, it is now symmetrical:
        # if the main speaker talks after another speaker (or is the first to talk), BOS is prepended to the first word.
        # Similary, if any other speaker speaks either first, or after the main speaker, a EOS is prepended.
        # This is in contrast with the legacy Interleaver, where the EOS would be inserted immediately
        # at the end of the turn of the main speaker.
        out: list[TokenizedAlignment] = []
        last_speaker = None
        for toks, ts, speaker in alignments:
            toks = list(toks)
            if speaker == last_speaker:
                pass
            elif speaker == main_speaker:
                toks.insert(0, self.tokenizer.bos_id())
            elif last_speaker == main_speaker:
                assert out
                toks.insert(0, self.tokenizer.eos_id())
            last_speaker = speaker
            out.append((toks, ts, speaker))
        return out

    def build_token_stream(
        self,
        alignments: list[TokenizedAlignment] | None,
        segment_duration: float,
    ) -> torch.Tensor:
        """Builds the token stream from the tokenized alignments."""
        T = math.ceil(segment_duration * self.audio_frame_rate)
        if alignments is None:
            text_tokens = [self.zero_padding] * T
        else:
            text_tokens = [self.text_padding] * T
            i = 0
            to_append_stack: deque = deque()
            last_word_end = -1
            for t in range(T):
                while (
                    i < len(alignments)
                    and alignments[i][1][0] * self.audio_frame_rate < t + 1
                ):
                    tokenized = alignments[i][0]
                    last_word_end = int(alignments[i][1][1] * self.audio_frame_rate)
                    if self.keep_and_shift:
                        to_append_stack.extend(tokenized)
                    else:
                        to_append_stack = deque(tokenized)
                    i += 1
                if to_append_stack:
                    if t > 0 and text_tokens[t - 1] in [
                        self.text_padding,
                        self.in_word_padding,
                    ]:
                        text_tokens[t - 1] = self.end_of_text_padding
                    next_token = to_append_stack.popleft()
                    text_tokens[t] = next_token
                elif t <= last_word_end:
                    text_tokens[t] = self.in_word_padding
        if self.audio_delay < 0:
            prefix_length = int(self.audio_frame_rate * -self.audio_delay)
            text_tokens[:prefix_length] = [self.zero_padding] * prefix_length
        return torch.tensor(text_tokens, device=self.device).view(1, 1, -1)

    def prepare_item(
        self,
        alignments: list[Alignment] | None,
        segment_duration: float,
        main_speaker: str | None = None,
    ) -> torch.Tensor:
        """Responsible with processing the alignments and calling `build_token_stream`."""
        if alignments is None:
            tokenized = None
        else:
            tokenized = self._tokenize(sorted(alignments, key=lambda x: x[1][0]))
            if self.keep_main_only:
                main_speaker = main_speaker or self.main_speaker_label
                tokenized = self._keep_main_only(tokenized, main_speaker)
            elif self.use_bos_eos:
                main_speaker = main_speaker or self.main_speaker_label
                tokenized = self._insert_bos_eos(tokenized, main_speaker)
            tokenized = self._keep_those_with_duration(tokenized)
            if self.audio_delay != 0:
                tokenized = self._add_delay(tokenized)
        return self.build_token_stream(tokenized, segment_duration)


def dicho(alignment, val, i=0, j=None):
    if j is None:
        j = len(alignment)
    if i == j:
        return i
    k = (i + j) // 2
    if alignment[k][1][0] < val:
        return dicho(alignment, val, k + 1, j)
    else:
        return dicho(alignment, val, i, k)


_FLAME_PATH_CACHE: dict[str, Optional[Path]] = {}  # stem → resolved FLAME .npy path (or None)


def _is_ami_stereo_path(path: str) -> bool:
    """Return True for the balanced AMI stereo dataset.

    AMI files are already stored in the model convention [target agent, user].
    For regular clips target agent=backchannel; for ``_switch_stereo`` clips
    target agent=utterance after the physical channel swap.
    """
    return any(
        part == "stereo_ami_balanced" or part.startswith("stereo_ami_balanced_")
        for part in Path(path).parts
    )


def _swap_vap_speaker_nibbles(labels: np.ndarray) -> np.ndarray:
    """Swap VAP spk0 bits [7:4] with spk1 bits [3:0]."""
    return ((labels & 0x0F) << 4) | ((labels >> 4) & 0x0F)


class InterleavedTokenizer:
    def __init__(
        self,
        mimi,
        interleaver,
        duration_sec: float,
        vap_manifest_path: str = "",
        flame_root: str = "",
        flame_speaker: str = "bc",
    ):
        """
        Args:
            mimi: Mimi audio tokenizer.
            interleaver: Text interleaver.
            duration_sec: Segment duration in seconds.
            vap_manifest_path: Path to VAP label manifest JSON (optional).
            flame_root: Root directory of FLAME .npy files, structured as
                ``{flame_root}/{speaker}/{split}/{stem}_{speaker}.npy``
                where speaker ∈ {"bc", "ut"} and split ∈ {"train", "valid", "test"}.
            flame_speaker: Primary speaker suffix for the agent channel ("bc" or "ut").
        """
        self.mimi = mimi
        self.interleaver = interleaver
        self.duration_sec = duration_sec
        self.num_audio_frames = math.ceil(duration_sec * mimi.frame_rate)
        self.mimi_sample_rate = mimi.sample_rate  # typically 24000

        # VAP manifest lookup tables
        self.vap_lookup = {}  # (file_id, vap_step_index) -> label_int (0-255, future window)
        self.vad_lookup = {}  # (file_id, vap_step_index) -> cur_va_int (0-3, current frame)
        self.vap_hop_s = 0.08  # default
        if vap_manifest_path and os.path.exists(vap_manifest_path):
            print(f"Loading VAP manifest from {vap_manifest_path}...")
            with open(vap_manifest_path, 'r') as f:
                manifest = json.load(f)
                self.vap_hop_s = manifest.get("config", {}).get("hop_duration_s", 0.08)
                for s in manifest.get("samples", []):
                    f_id = s["file_id"]
                    # Store by (file_id, vap_step_index); step = offset_seconds / hop_s
                    idx = int(round(s["offset_seconds"] / self.vap_hop_s))
                    self.vap_lookup[(f_id, idx)] = s["label_int"]
                    # Current-frame VAD, written by vap_window.py alongside the VAP
                    # label. Older manifests predate this field — the VAD target is
                    # then left unknown (-100) and its loss term is skipped.
                    cur_va = s.get("cur_va_int")
                    if cur_va is not None:
                        self.vad_lookup[(f_id, idx)] = cur_va
            print(f"Loaded {len(self.vap_lookup)} VAP targets (hop={self.vap_hop_s}s).")
            if self.vad_lookup:
                print(f"Loaded {len(self.vad_lookup)} current-frame VAD targets.")
            else:
                print("[WARNING] Manifest has no 'cur_va_int' field — VAD targets "
                      "unavailable. Regenerate the manifest with vap_window.py to "
                      "enable vad_loss.")

        # FLAME / 3DMM face motion configuration
        self.flame_root = flame_root
        self.flame_speaker = flame_speaker
        self._motion_fps = 25  # FLAME data frame rate

    def tokenize_with_encoded_audio(
        self,
        audio_tokens: torch.Tensor,
        start_sec: float,
        path: str,
        voice_prompt_emb: torch.Tensor | None = None,
        actual_wav_samples: int | None = None,
    ) -> Sample:
        """Finish tokenization given pre-encoded mimi tokens.

        audio_tokens: raw output of mimi.encode for one sample — shape [C, K, T_enc]
        where C = number of audio channels (typically 2 for stereo conversations).
        Called by the batched encoding path in data_loader so mimi.encode is invoked
        only once per batch rather than once per sample.
        """
        is_ami_stereo = _is_ami_stereo_path(path)
        if is_ami_stereo:
            if audio_tokens.shape[0] != 2:
                raise ValueError(
                    f"Expected two AMI stereo channels, got {audio_tokens.shape[0]}: {path}"
                )
            # AMI is already role-normalized: ch0=target agent, ch1=user.
            # Do not swap; flattened rows become 1:9=agent and 9:17=user.

        audio_tokens = audio_tokens[..., : self.num_audio_frames]
        this_num_audio_frames = audio_tokens.shape[-1]
        audio_tokens = torch.nn.functional.pad(
            audio_tokens[..., : self.num_audio_frames],
            (0, self.num_audio_frames - this_num_audio_frames),
            value=self.interleaver.zero_padding,
        )
        audio_tokens = audio_tokens.view(1, -1, self.num_audio_frames)

        info_file = os.path.splitext(path)[0] + ".json"
        with open(info_file) as f:
            data = json.load(f)
        alignments = data["alignments"]

        start_alignment = dicho(alignments, start_sec)
        end_alignment = dicho(alignments, start_sec + self.duration_sec)
        alignments = [
            (a[0], (a[1][0] - start_sec, a[1][1] - start_sec), a[2])
            for a in alignments[start_alignment:end_alignment]
        ]

        text_tokens = self.interleaver.prepare_item(alignments, this_num_audio_frames)
        text_tokens = torch.nn.functional.pad(
            text_tokens,
            (0, self.num_audio_frames - text_tokens.shape[-1]),
            value=self.interleaver.zero_padding,
        )

        codes = torch.cat([text_tokens, audio_tokens], dim=1)

        # --- VAP / VAD Target Processing ---
        vap_targets = None
        vad_targets = None
        if self.vap_lookup:
            vap_targets = torch.full((1, self.num_audio_frames), -100, dtype=torch.long, device=codes.device)
            if self.vad_lookup:
                # Allocated unconditionally, mirroring vap_targets above: unmatched
                # frames stay -100 and are dropped by the BCE mask at loss time.
                # Leaving this None for an unmatched file instead would make
                # Batch.collate's all() discard the targets for the WHOLE batch —
                # at a ~93% per-sample match rate that silently disables vad_loss
                # for all but 1 batch in ~11,000.
                vad_targets = torch.full(
                    (1, 2, self.num_audio_frames), -100.0, dtype=torch.float32
                )
            raw_file_id = os.path.splitext(os.path.basename(path))[0]
            is_ami_switched = (
                is_ami_stereo and raw_file_id.endswith("_switch_stereo")
            )

            # DualTalk: the two stereo files of a conversation are named
            # "{sid}_speaker" and "{sid}_listener" and share a single VAP manifest
            # entry "{sid}_speaker1_cleaned" (generated with spk0=speaker1, spk1=speaker2).
            # The training channel convention is ch0=agent(spk1), ch1=user(spk0), and by
            # audio correlation "_speaker" has ch0=speaker1 while "_listener" has ch0=speaker2:
            #   - "_speaker"  → agent=speaker1 ≠ manifest spk1 → reverse the label (swap spk0/spk1)
            #   - "_listener" → agent=speaker2 = manifest spk1 → keep the label as-is
            dt_reverse = False
            if raw_file_id.endswith("_speaker"):
                file_id_candidates = [raw_file_id[: -len("_speaker")] + "_speaker1_cleaned"]
                dt_reverse = True
            elif raw_file_id.endswith("_listener"):
                file_id_candidates = [raw_file_id[: -len("_listener")] + "_speaker1_cleaned"]
            elif is_ami_switched:
                # The switched AMI file reuses the original pair's VAP labels.
                # Example:
                #   audio:    EN2001a_0_switch_stereo.wav
                #   manifest: EN2001a_0_ut
                original_base = raw_file_id[: -len("_switch_stereo")]
                file_id_candidates = [
                    original_base + "_ut",
                    original_base,
                ]
            else:
                base = raw_file_id.replace("_stereo", "")
                file_id_candidates = [raw_file_id, base + "_ut", base]

            # Whether manifest spk0/spk1 must be swapped to match the training
            # convention (spk0=user, spk1=agent). Drives BOTH the VAP nibble swap
            # and the VAD row order below, so the two labels stay consistent.
            #   dt_reverse                        : DualTalk "_speaker" — agent=speaker1 ≠ manifest spk1
            #   is_ami_stereo & not is_ami_switched: AMI manifest spk0=backchannel agent, spk1=user
            # Switched AMI needs no swap: after its physical channel swap,
            # utterance=agent and backchannel=user, already matching the manifest.
            do_swap = dt_reverse or (is_ami_stereo and not is_ami_switched)

            moshi_frame_duration = 1.0 / self.mimi.frame_rate
            t_arr = np.arange(self.num_audio_frames, dtype=np.float64)
            vap_steps = np.rint(
                (start_sec + t_arr * moshi_frame_duration) / self.vap_hop_s
            ).astype(np.int64)

            matched_fid = None
            for fid in file_id_candidates:
                labels = np.array(
                    [self.vap_lookup.get((fid, int(s)), -100) for s in vap_steps],
                    dtype=np.int64,
                )
                valid = labels != -100
                if valid.any():
                    if do_swap:
                        # Swap spk0 (label bits 7-4) and spk1 (bits 3-0), preserving the
                        # within-speaker bin order (bin0 = nearest 200ms stays the MSB of its nibble).
                        rev = _swap_vap_speaker_nibbles(labels)
                        labels = np.where(valid, rev, labels)
                    vap_targets[0, valid] = torch.from_numpy(labels[valid]).to(codes.device)
                    matched_fid = fid
                    break

            # if matched_fid is None:
            #     print(f"[WARNING] No VAP target found for '{raw_file_id}' "
            #           f"All targets set to -100.")

            # ── Current-frame VAD targets (v2: vad_logits supervision) ────────
            # Read straight from the manifest (cur_va_int, written by vap_window.py)
            # instead of re-running an energy VAD on the waveform here. The manifest
            # VAD is computed after per-channel RMS normalisation, so quiet speakers
            # are handled consistently, and it shares the exact frame grid used for
            # the VAP labels (hop 0.08s == one mimi frame @ 12.5 Hz).
            #
            # cur_va_int = (spk0 << 1) | spk1. Output row order is (user, agent) to
            # match vad_logits ([..., 0] = user from out["x1"], [..., 1] = agent from
            # out["x2"]); do_swap decides which manifest speaker is which.
            # Unknown frames are -100 and are dropped from the BCE at loss time.
            if self.vad_lookup and matched_fid is not None:
                cur = np.array(
                    [self.vad_lookup.get((matched_fid, int(s)), -1) for s in vap_steps],
                    dtype=np.int64,
                )
                known = cur >= 0
                if known.any():
                    spk0 = (cur >> 1) & 1
                    spk1 = cur & 1
                    user, agent = (spk1, spk0) if do_swap else (spk0, spk1)
                    va_full = np.full((2, self.num_audio_frames), -100.0, dtype=np.float32)
                    va_full[0, known] = user[known]
                    va_full[1, known] = agent[known]
                    vad_targets = torch.from_numpy(va_full).unsqueeze(0)  # [1, 2, T]
            # else: vad_targets keeps its all -100 fill — the sample contributes no
            # VAD frames, but the rest of the batch still does.

        # ── Binary BC timing target (vectorized) ─────────────────────────────
        # Agent (spk1) speaks in next 200ms AND user (spk0) is silent.
        # Derived from the same VAP manifest labels (no extra data needed):
        #   spk0_bin0 = bit 7 of label_int,  spk1_bin0 = bit 3 of label_int
        # Base rate: ~3% — use focal loss with pos_weight at training time.
        bc_timing_targets = None
        if self.vap_lookup and vap_targets is not None:
            lbl = vap_targets[0]  # [T] long tensor with -100 for unknown
            valid = lbl != -100
            bc = torch.full((self.num_audio_frames,), -100.0, dtype=torch.float32, device=codes.device)
            if valid.any():
                lbl_valid = lbl[valid].long()
                spk0_bin0 = (lbl_valid >> 7) & 1   # user speaking in next 200ms
                spk1_bin0 = (lbl_valid >> 3) & 1   # agent speaking in next 200ms
                bc[valid] = ((spk1_bin0 == 1) & (spk0_bin0 == 0)).float()
            bc_timing_targets = bc.unsqueeze(0)  # [1, T]

        # ── FLAME / 3DMM face motion ─────────────────────────────────────────
        face_motion_gt = None
        if self.flame_root:
            face_motion_gt = self._load_face_motion(path, start_sec)

        # ── Silence-padding validity mask ─────────────────────────────────────
        # When a short clip is zero-padded to batch max_T before mimi encoding,
        # the padded region produces silence mimi tokens that should be excluded
        # from all losses (audio, text, VAP, face).
        valid_mask = None
        valid_face_frames = None
        if actual_wav_samples is not None:
            actual_mimi_frames = min(
                math.ceil(actual_wav_samples * self.mimi.frame_rate / self.mimi_sample_rate),
                self.num_audio_frames,
            )
            valid_mask = torch.zeros(self.num_audio_frames, dtype=torch.bool)
            valid_mask[:actual_mimi_frames] = True

            n_face_frames = int(self.duration_sec * self._motion_fps)
            # Use ceil to be consistent with actual_mimi_frames above.
            # floor would exclude the last 1–2 face frames that correspond to
            # the last valid mimi frame (since face_fps = 2 × mimi_fps).
            valid_face_frames = min(
                math.ceil(actual_wav_samples / self.mimi_sample_rate * self._motion_fps),
                n_face_frames,
            )

        return Sample(codes, data.get("text_conditions", None), vap_targets=vap_targets,
                      voice_prompt_emb=voice_prompt_emb, face_motion_gt=face_motion_gt,
                      valid_mask=valid_mask, valid_face_frames=valid_face_frames,
                      vad_targets=vad_targets,
                      bc_timing_targets=bc_timing_targets)

    def _resolve_flame_target(self, audio_path: str) -> tuple[str, str, bool]:
        """Resolve FLAME stem/speaker for one audio sample.

        Returns ``(stem, speaker, exact)``. In AMI auto mode, the original
        sample trains the backchannel agent (bc), while a channel-switched
        sample trains the utterance agent (ut). The switched FLAME arrays reuse
        the original pair stem, so the synthetic ``_switch`` suffix is removed.
        ``exact=True`` prevents falling back to the other person's motion.
        """
        stem = os.path.splitext(os.path.basename(audio_path))[0]
        if stem.endswith("_stereo"):
            stem = stem[: -len("_stereo")]

        if self.flame_speaker == "auto":
            if stem.endswith("_switch"):
                return stem[: -len("_switch")], "ut", True
            return stem, "bc", True
        return stem, self.flame_speaker, False

    def _find_flame_path(
        self, stem: str, speaker: Optional[str] = None, exact: bool = False
    ) -> Optional[Path]:
        """Return the FLAME .npy path for *stem*, or None if not found.

        Search order: *speaker* override (per-file agent) when given, else the
        configured primary speaker then fallback speaker.
        For each speaker, tries subdirectory patterns:
          {flame_root}/{speaker}/{split}/{stem}_{speaker}.npy
          {flame_root}/{split}/{stem}_{speaker}.npy   (flat layout)
        Results are cached per (stem, speaker) pair.
        """
        primary_speaker = speaker or self.flame_speaker
        cache_key = f"{stem}|{primary_speaker}|exact={exact}"
        if cache_key in _FLAME_PATH_CACHE:
            return _FLAME_PATH_CACHE[cache_key]

        root = Path(self.flame_root)
        if exact:
            speakers = [primary_speaker]
        else:
            fallback = "ut" if primary_speaker == "bc" else "bc"
            speakers = [primary_speaker, fallback]
        splits = ["train", "valid", "val", "test"]
        result: Optional[Path] = None

        for speaker in speakers:
            target = f"{stem}_{speaker}"
            for split in splits:
                # Standard layout: {root}/{speaker}/{split}/{target}.npy
                p = root / speaker / split / f"{target}.npy"
                if p.exists():
                    result = p
                    break
                # Flat layout: {root}/{split}/{target}.npy
                p = root / split / f"{target}.npy"
                if p.exists():
                    result = p
                    break
            if result is not None:
                break

        _FLAME_PATH_CACHE[cache_key] = result
        return result

    def _load_face_motion(self, audio_path: str, start_sec: float) -> Optional[torch.Tensor]:
        """Load the FLAME motion window aligned to this audio segment.

        Returns a float32 tensor of shape [T_face, 54] where T_face = duration_sec * 25,
        or None when no matching FLAME file is found.
        """
        n_face_frames = int(self.duration_sec * self._motion_fps)
        start_frame = int(start_sec * self._motion_fps)

        stem, speaker, exact = self._resolve_flame_target(audio_path)
        flame_path = self._find_flame_path(stem, speaker=speaker, exact=exact)
        if flame_path is None:
            return None

        try:
            arr = np.load(flame_path, allow_pickle=True)
            if arr.dtype == object:
                # Array of per-frame dicts → parse each frame
                motion_list = []
                for frame_idx, frame in enumerate(arr):
                    expr = np.asarray(frame.get("expr", []), dtype=np.float32)
                    if expr.shape[0] < 50:
                        raise ValueError(
                            f"FLAME frame {frame_idx} has only {expr.shape[0]} expression "
                            f"coefficients (expected >=50): {flame_path}"
                        )
                    expr = expr[:50]
                    jaw_raw = np.asarray(frame.get("jaw_pose", []), dtype=np.float32)
                    neck_raw = np.asarray(frame.get("neck_pose", []), dtype=np.float32)
                    if jaw_raw.shape[0] < 1 or neck_raw.shape[0] < 3:
                        raise ValueError(
                            f"FLAME frame {frame_idx} is missing jaw/neck pose values: "
                            f"{flame_path}"
                        )
                    jaw = jaw_raw[:1]
                    neck = neck_raw[:3]
                    motion_list.append(np.concatenate([expr, jaw, neck]))
                motion = np.stack(motion_list, axis=0).astype(np.float32)
            else:
                motion = np.asarray(arr, dtype=np.float32)
                if motion.shape[-1] == 56:
                    # 56-dim (expr50 + jaw3 + neck3) → 54-dim (expr50 + jaw1 + neck3)
                    motion = np.concatenate([motion[..., :50], motion[..., 50:51], motion[..., 53:56]], axis=-1)
                elif motion.shape[-1] != 54:
                    raise ValueError(
                        f"Unsupported FLAME dimension {motion.shape[-1]} (expected 54 or 56): "
                        f"{flame_path}"
                    )
        except Exception as exc:
            logger.warning(
                "[FLAME] Skipping sample because target could not be read: "
                f"audio={audio_path}, flame={flame_path}, error={type(exc).__name__}: {exc}"
            )
            return None

        if motion.ndim != 2 or motion.shape[-1] != 54:
            logger.warning(
                "[FLAME] Skipping sample with invalid target shape: "
                f"audio={audio_path}, flame={flame_path}, shape={motion.shape}, expected=[T,54]"
            )
            return None
        if not np.isfinite(motion).all():
            logger.warning(
                "[FLAME] Skipping sample because target contains NaN/Inf: "
                f"audio={audio_path}, flame={flame_path}"
            )
            return None

        # Slice temporal window matching the audio segment.
        end_frame = start_frame + n_face_frames
        if start_frame >= motion.shape[0]:
            logger.warning(
                "[FLAME] Skipping out-of-range segment: "
                f"audio={audio_path}, flame={flame_path}, start={start_frame}, "
                f"available_frames={motion.shape[0]}"
            )
            return None
        window = motion[start_frame:end_frame]
        if window.shape[0] < n_face_frames:
            pad = np.zeros((n_face_frames - window.shape[0], 54), dtype=np.float32)
            window = np.concatenate([window, pad], axis=0)
        return torch.from_numpy(window).float()  # [T_face, 54]

    def has_flame_file(self, path: str) -> bool:
        """Return True if a FLAME .npy file exists for this audio path.

        Uses the same cached lookup as _load_face_motion.  Always returns True
        when flame_root is not configured (face gen disabled for this tokenizer).
        """
        if not self.flame_root:
            return True
        stem, speaker, exact = self._resolve_flame_target(path)
        return self._find_flame_path(stem, speaker=speaker, exact=exact) is not None

    def __call__(self, wav: np.ndarray, start_sec: float, path: str, voice_prompt_emb: torch.Tensor | None = None) -> Sample:
        with torch.no_grad():
            audio_tensor = torch.Tensor(wav).cuda()
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)  # [T] -> [1, T]
            audio_tokens = self.mimi.encode(audio_tensor[:, None])
        return self.tokenize_with_encoded_audio(audio_tokens, start_sec, path, voice_prompt_emb)
