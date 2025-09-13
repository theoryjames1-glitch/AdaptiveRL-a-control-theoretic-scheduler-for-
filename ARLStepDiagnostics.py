# ARLStepDiagnostics.py
# Per-step diagnostics with a configurable logger for the stepwise AdaptiveRL controller
# No sigils, No recursions, No glyphs, No rituals

import os, json, time, math
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from transformers import TrainerCallback

# ----------------------------
# Config
# ----------------------------

@dataclass
class ARLStepDiagConfig:
    # File output
    fname: str = "arl_step_tape.jsonl"    # JSONL path under output_dir
    flush_every: int = 1                  # flush every N writes
    max_lines: Optional[int] = None       # rollover after N lines (None = no rollover)
    rollover_suffix: str = ".part{idx}"   # appended to fname when rolling over

    # Console echo
    echo_every: int = 1                   # print every N logs (1 = every log)
    echo_prefix: str = "[STEP]"
    echo_columns: List[str] = field(default_factory=lambda: [
        "step", "loss", "sched_lr", "opt_lr", "alpha", "mu", "sigma"
    ])
    # Column formatting map (value formatters by key)
    fmt: Dict[str, str] = field(default_factory=lambda: {
        "step": "{:6d}",
        "epoch": "{:6.2f}",
        "loss": "{:>10}",
        "grad_norm": "{:>8}",
        "sched_lr": "{:g}",
        "opt_lr": "{:g}",
        "alpha": "{:.6g}",
        "mu": "{:.4f}",
        "sigma": "{:.3g}",
        "beta1_now": "{:.4f}",
    })

    # What to record
    log_optimizer: bool = True            # record actual optimizer lr/beta1
    log_scheduler_lr: bool = True         # record HF 'learning_rate' from logs
    log_controller: bool = True           # record alpha/mu/sigma
    log_internals: bool = False           # record controller internals (_dL/_dR/EMAs/plateau)
    # Optional custom extra fields (callable or static)
    extras: Optional[Dict[str, Any]] = None


# ----------------------------
# Diagnostics Callback
# ----------------------------

class ARLStepDiagnostics(TrainerCallback):
    """
    Logs one JSON line at every Trainer `on_log` (i.e., every `logging_steps`).
    Also prints a concise console line per `cfg.echo_every`.

    Recorded fields (toggle via config):
      - step, epoch, loss, grad_norm
      - sched_lr (from HF logs), opt_lr (actual optimizer value), beta1_now
      - controller alpha/mu/sigma
      - (optional) controller internals: dL/dR, el/er, vl, scale EMAs, plateau

    Usage:
        from ARLStepDiagnostics import ARLStepDiagnostics, ARLStepDiagConfig
        cfg = ARLStepDiagConfig(echo_every=1, log_internals=False)
        trainer.add_callback(ARLStepDiagnostics(controller, out_dir=args.output_dir, cfg=cfg, trainer=trainer))
    """
    def __init__(self, controller, out_dir: str, cfg: Optional[ARLStepDiagConfig] = None, trainer=None):
        self.ctrl = controller
        self.cfg = cfg or ARLStepDiagConfig()
        self._trainer = trainer  # optional pre-bound trainer for older HF
        self._path_base = os.path.join(out_dir, self.cfg.fname)
        os.makedirs(out_dir, exist_ok=True)
        self._file = None
        self._open_sink(self._path_base)
        self._echo_count = 0
        self._write_count = 0
        self._part_idx = 0

    # ---- lifecycle ----

    def on_train_begin(self, args, state, control, **kw):
        # Emit a meta record with controller config (if available)
        meta = {
            "type": "meta",
            "t": time.time(),
            "alpha_bounds": getattr(getattr(self.ctrl, "cfg", None), "alpha_bounds", None),
            "mu_bounds": getattr(getattr(self.ctrl, "cfg", None), "mu_bounds", None),
            "sigma_bounds": getattr(getattr(self.ctrl, "cfg", None), "sigma_bounds", None),
            "gains": getattr(getattr(self.ctrl, "cfg", None), "gains", None),
            "ema_betas": getattr(getattr(self.ctrl, "cfg", None), "ema_betas", None),
            "update_every": getattr(getattr(self.ctrl, "cfg", None), "update_every", None),
            "lr_change_cap": getattr(getattr(self.ctrl, "cfg", None), "lr_change_cap", None),
            "plateau_eps_d": getattr(getattr(self.ctrl, "cfg", None), "plateau_eps_d", None),
            "plateau_eps_v": getattr(getattr(self.ctrl, "cfg", None), "plateau_eps_v", None),
        }
        self._write_json(meta)

    def on_log(self, args, state, control, logs=None, **kw):
        if not logs:
            return
        tr = self._get_trainer(kw)

        # Core fields
        rec = {
            "t": time.time(),
            "step": int(state.global_step or 0),
            "epoch": float(state.epoch or 0.0),
            "loss": _maybe_num(logs.get("loss")),
            "grad_norm": _maybe_num(logs.get("grad_norm")),
        }

        # Scheduler LR from HF logs
        if self.cfg.log_scheduler_lr:
            rec["sched_lr"] = _maybe_num(logs.get("learning_rate"))

        # Actual optimizer LR and beta1
        if self.cfg.log_optimizer and tr is not None and getattr(tr, "optimizer", None) is not None:
            try:
                pg0 = tr.optimizer.param_groups[0]
                rec["opt_lr"] = _maybe_num(pg0.get("lr"))
                if "betas" in pg0:
                    rec["beta1_now"] = _maybe_num(pg0["betas"][0])
            except Exception:
                pass

        # Controller surface signals
        if self.cfg.log_controller and self.ctrl is not None:
            rec["alpha"] = _maybe_num(getattr(self.ctrl, "alpha", None))
            rec["mu"]    = _maybe_num(getattr(self.ctrl, "mu", None))
            rec["sigma"] = _maybe_num(getattr(self.ctrl, "sigma", None))

        # Optional internals (if the step controller exposes them)
        if self.cfg.log_internals and self.ctrl is not None:
            for k in ("el","er","vl","Sdl","Sdr","Sv","_dL","_dR"):
                rec[k] = _maybe_num(getattr(self.ctrl, k, None))
            # recompute plateau for visibility if present
            try:
                rec["plateau"] = _plateau_from_ctrl(self.ctrl)
            except Exception:
                pass

        # Custom extras (static values or callables)
        if isinstance(self.cfg.extras, dict):
            for k, v in self.cfg.extras.items():
                rec[k] = v(self) if callable(v) else v

        self._write_json(rec)

        # Echo
        self._echo_count += 1
        if (self._echo_count % max(1, self.cfg.echo_every)) == 0:
            print(self._format_echo_line(rec))

    def on_train_end(self, args, state, control, **kw):
        self._close_sink()

    # ---- helpers ----

    def _get_trainer(self, kwargs):
        return kwargs.get("trainer", self._trainer)

    def _open_sink(self, path):
        # roll if needed
        if self._file is not None:
            self._close_sink()
        self._file = open(path, "a", buffering=1)
        self._write_count = 0

    def _close_sink(self):
        try:
            if self._file:
                self._file.flush()
                self._file.close()
        finally:
            self._file = None

    def _rollover_if_needed(self):
        if self.cfg.max_lines is None:
            return
        if self._write_count >= int(self.cfg.max_lines):
            self._part_idx += 1
            base, ext = os.path.splitext(self._path_base)
            new_path = f"{base}{self.cfg.rollover_suffix.format(idx=self._part_idx)}{ext}"
            self._open_sink(new_path)

    def _write_json(self, obj: Dict[str, Any]):
        line = json.dumps({k: v for k, v in obj.items() if v is not None})
        self._file.write(line + "\n")
        self._write_count += 1
        if self.cfg.flush_every and (self._write_count % int(self.cfg.flush_every) == 0):
            try:
                self._file.flush()
            except Exception:
                pass
        self._rollover_if_needed()

    def _format_echo_line(self, rec: Dict[str, Any]) -> str:
        cols = self.cfg.echo_columns or []
        parts = []
        for key in cols:
            val = rec.get(key, None)
            if val is None:
                continue
            fmt = self.cfg.fmt.get(key, "{}")
            try:
                parts.append(f"{key}={fmt.format(val)}")
            except Exception:
                parts.append(f"{key}={val}")
        return f"{self.cfg.echo_prefix} " + "  ".join(parts)


# ----------------------------
# Utilities
# ----------------------------

def _maybe_num(x):
    if x is None:
        return None
    try:
        if isinstance(x, (int, float)):
            return float(x)
        # strings from some loggers
        return float(str(x))
    except Exception:
        return x

def _plateau_from_ctrl(ctrl) -> Optional[float]:
    # Mirrors the controller plateau formula if internals exist
    # P = sigm((eps_d - |dL|)/0.25) * sigm((eps_v - (vl/Sv))/0.25)
    try:
        dL = abs(float(getattr(ctrl, "_dL")))
        vl = float(getattr(ctrl, "vl"))
        Sv = float(getattr(ctrl, "Sv")) + 1e-8
        eps_d = getattr(getattr(ctrl, "cfg"), "plateau_eps_d", 0.05)
        eps_v = getattr(getattr(ctrl, "cfg"), "plateau_eps_v", 0.02)
        s = lambda z: 1.0 / (1.0 + math.exp(-z))
        a = s((eps_d - dL) / 0.25)
        b = s((eps_v - (vl / Sv)) / 0.25)
        return float(a * b)
    except Exception:
        return None
