"""Post-optimizer scalar monitoring and a graceful step-500 gate (single GPU)."""
from __future__ import annotations

import json
import math
import statistics
from pathlib import Path


WEIGHTS = ("bc", "vap", "vad")
SCALARS = (*WEIGHTS, "bias")


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def trajectory_summary(history: list[dict], policy) -> dict:
    """History contains EVERY post-update step, including step zero."""
    gate = [r for r in history if r["step"] <= policy.gate_step]
    late = [r for r in gate if policy.gate_step - 100 <= r["step"]]
    end = [r for r in gate if policy.gate_step - 50 <= r["step"]]
    first = history[0]["weights"]
    excursion, late_range, end_shift = {}, {}, {}
    for key in WEIGHTS:
        excursion[key] = max(abs(r["weights"][key] - first[key]) for r in gate)
        late_range[key] = max(r["weights"][key] for r in late) - min(r["weights"][key] for r in late)
        end_shift[key] = abs(statistics.median(r["weights"][key] for r in end) - first[key])
    dynamic = {k: excursion[k] >= policy.dynamic_excursion and
               end_shift[k] >= policy.dynamic_end_shift for k in WEIGHTS}
    static = all(excursion[k] < policy.static_excursion and
                 late_range[k] < policy.static_late_range for k in WEIGHTS)
    result = {"E500": excursion, "Rlate": late_range, "Dend": end_shift,
              "dynamic": dynamic, "N_dynamic": sum(dynamic.values()),
              "min_E500": min(excursion.values()), "static": static,
              "gate_observed": history[-1]["step"] >= policy.gate_step,
              "initial_weights": first, "final_weights": history[-1]["weights"],
              "tail_drift": None, "tail_range": None, "stable": False,
              "Q800": {}, "Q1000": {}}
    if history[-1]["step"] >= 1000:
        early_tail = [r for r in history if 800 <= r["step"] <= 850]
        final_tail = [r for r in history if 950 <= r["step"] <= 1000]
        tail = [r for r in history if 900 <= r["step"] <= 1000]
        result["Q800"] = {k: statistics.median(r["weights"][k] for r in early_tail) for k in WEIGHTS}
        result["Q1000"] = {k: statistics.median(r["weights"][k] for r in final_tail) for k in WEIGHTS}
        result["tail_drift"] = max(abs(result["Q1000"][k] - result["Q800"][k]) for k in WEIGHTS)
        result["tail_range"] = max(max(r["weights"][k] for r in tail) -
                                   min(r["weights"][k] for r in tail) for k in WEIGHTS)
        result["stable"] = max(result["tail_drift"], result["tail_range"]) <= policy.stable_tail
    return result


class FusionDynamicsMonitor:
    def __init__(self, model, optimizer, args, run_dir: Path):
        self.policy = args.fusion_dynamics
        self.root = run_dir
        self.params = {}
        for name, param in model.named_parameters():
            for key in SCALARS:
                suffix = "backchannel_fusion_bias" if key == "bias" else f"backchannel_fusion_{key}_weight"
                if name.split(".")[-1] == suffix:
                    self.params[key] = param
        if set(self.params) != set(SCALARS):
            raise RuntimeError("Fusion monitor could not find all four scalar parameters")
        fusion_group = next((g for g in optimizer.param_groups if g.get("group_name") == "fusion"), None)
        if fusion_group is None:
            raise RuntimeError("Fusion optimizer group is missing")
        member_ids = {id(p) for p in fusion_group["params"]}
        for key, param in self.params.items():
            if param.numel() != 1 or not param.requires_grad or id(param) not in member_ids:
                raise RuntimeError(f"Invalid trainable fusion scalar: {key}")
        self.optimizer = optimizer
        self.args = args
        self.history = []
        self.nonzero = dict.fromkeys(SCALARS, 0)
        self.grad_nonzero = dict.fromkeys(SCALARS, 0)
        self.stalled = dict.fromkeys(SCALARS, 0)
        self.validation = {}
        self.label_totals = dict.fromkeys(WEIGHTS, 0)
        self.pre_grad = {}
        initial = self.values()
        for key in SCALARS:
            expected = getattr(args.backchannel, f"fusion_{key}_init")
            # Config values such as 0.9 are rounded to BF16 at initialization.
            if abs(initial[key] - expected) > 0.005:
                raise RuntimeError(f"Fusion init was overwritten: {key}: {initial[key]} vs {expected}")
        self.history.append({"step": 0, "weights": initial})
        self.append({"step": 0, "weights": initial,
                     "dtype": {k: str(p.dtype) for k, p in self.params.items()},
                     "requires_grad": {k: p.requires_grad for k, p in self.params.items()},
                     "policy": vars(self.policy)})
        self.write_status("running", 0)

    def values(self):
        values = {k: float(p.detach().float().item()) for k, p in self.params.items()}
        if not all(math.isfinite(v) for v in values.values()):
            self.fail("failed_nonfinite", "Nonfinite fusion scalar")
        return values

    def gradients(self):
        return {k: None if p.grad is None else float(p.grad.detach().float().item())
                for k, p in self.params.items()}

    def before_clip(self):
        self.pre_grad = self.gradients()
        if any(v is not None and not math.isfinite(v) for v in self.pre_grad.values()):
            self.fail("failed_nonfinite", "Nonfinite fusion gradient before clipping")

    def before_update(self):
        self.before = self.values()
        self.post_grad = self.gradients()
        self.used_lrs = {g.get("group_name", str(i)): float(g["lr"])
                         for i, g in enumerate(self.optimizer.param_groups)}

    def after_update(self, step: int, aux: dict):
        for key, values in aux.items():
            self.label_totals[key] += values.get("valid_label_count", 0)
            if not math.isfinite(values["raw"]):
                self.fail("failed_nonfinite", f"Nonfinite raw auxiliary loss: {key}")
        weights = self.values()
        delta = {k: weights[k] - self.before[k] for k in SCALARS}
        for key in SCALARS:
            self.nonzero[key] += delta[key] != 0
            gradient = self.post_grad[key]
            if gradient is not None and not math.isfinite(gradient):
                self.fail("failed_nonfinite", f"Nonfinite gradient: {key}")
            active = gradient is not None and gradient != 0
            self.grad_nonzero[key] += active
            self.stalled[key] += active and delta[key] == 0
        self.history.append({"step": step, "weights": weights})
        if step % self.args.log_freq == 0 or step == self.policy.gate_step:
            self.append({"step": step, "weights": weights, "delta": delta,
                         "grad_before_clip": self.pre_grad, "grad_after_clip": self.post_grad,
                         "lr": self.used_lrs, "auxiliary": aux,
                         "nonzero_update_fraction": {k: v / step for k, v in self.nonzero.items()}})
        if step == self.policy.gate_step:
            self.gate = trajectory_summary(self.history, self.policy)
            atomic_json(self.root / "fusion_gate_500.json", self.gate)
            return self.gate["static"]
        return False

    def record_validation(self, step: int, text_loss, evidence):
        if text_loss is None or not math.isfinite(text_loss):
            self.fail("monitor_error", "Missing/nonfinite validation text loss")
        if evidence.get("fusion/evidence_frames", 0) <= 0:
            self.fail("monitor_error", "Fixed validation batch has no fusion evidence frames")
        self.validation[str(step)] = {"text_loss": float(text_loss), "evidence": evidence}
        self.append({"step": step, "validation": self.validation[str(step)]}, "fusion_validation.jsonl")

    def append(self, record, filename="fusion_trajectory.jsonl"):
        with (self.root / filename).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, allow_nan=False) + "\n")

    def write_status(self, status, step, **extra):
        atomic_json(self.root / "fusion_status.json", {"status": status, "step": step, **extra})

    def fail(self, status, reason):
        step = self.history[-1]["step"] if self.history else 0
        self.write_status(status, step, reason=reason)
        raise RuntimeError(f"Fusion dynamics {status}: {reason}")

    def finish(self, step: int, stopped: bool):
        for key, coefficient_name in (("bc", "bc_event_loss_weight"),
                                      ("vap", "vap_loss_weight"), ("vad", "vad_loss_weight")):
            if getattr(self.args.backchannel, coefficient_name, 0) > 0 and self.label_totals[key] == 0:
                self.fail("monitor_error", f"Active {key} auxiliary loss had no valid labels")
        summary = trajectory_summary(self.history, self.policy)
        summary["validation"] = self.validation
        summary["valid_label_totals"] = self.label_totals
        summary["update_counts"] = self.nonzero
        summary["gradient_nonzero_counts"] = self.grad_nonzero
        summary["rounding_stall_counts"] = self.stalled
        summary["precision_stalled"] = all(self.grad_nonzero[k] > 0 and self.nonzero[k] == 0 for k in WEIGHTS)
        summary["movement_class"] = ("static" if summary["static"] else
                                      "fully_dynamic" if summary["N_dynamic"] == 3 else "partially_dynamic")
        status = "precision_stalled" if stopped and summary["precision_stalled"] else (
            "early_stopped_static" if stopped else "completed")
        summary.update(status=status, step=step)
        checkpoint = self.root / "checkpoints" / f"checkpoint_{step:06d}" / "consolidated" / "consolidated.safetensors"
        if not checkpoint.is_file():
            self.fail("monitor_error", f"Final checkpoint was not saved: {checkpoint}")
        from safetensors import safe_open
        with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
            for key in SCALARS:
                name = "backchannel_fusion_bias" if key == "bias" else f"backchannel_fusion_{key}_weight"
                names = [n for n in handle.keys() if n.split(".")[-1] == name]
                if len(names) != 1 or float(handle.get_tensor(names[0]).float().item()) != summary["final_weights"][key]:
                    self.fail("monitor_error", f"Checkpoint/trajectory mismatch: {key}")
        summary["checkpoint"] = str(checkpoint)
        atomic_json(self.root / "fusion_summary.json", summary)
        self.write_status(status, step, summary=str(self.root / "fusion_summary.json"))
