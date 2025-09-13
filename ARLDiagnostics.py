# ARLDiagnostics.py
# No sigils, No recursions, No glyphs, No rituals

import json, os, math, time
from transformers import TrainerCallback

class ARLDiagnostics(TrainerCallback):
    """
    Writes a JSONL tape of controller internals per evaluation and logs scalars
    into the Trainer's logging stream (so they show up in TB/W&B if enabled).
    """
    def __init__(self, controller, out_dir: str, fname: str = "arl_tape.jsonl"):
        self.ctrl = controller
        self.path = os.path.join(out_dir, fname)
        os.makedirs(out_dir, exist_ok=True)
        self._f = open(self.path, "a", buffering=1)

    def on_train_begin(self, args, state, control, **kw):
        # Header record
        self._f.write(json.dumps({
            "type": "meta",
            "time": time.time(),
            "alpha_bounds": self.ctrl.cfg.alpha_bounds,
            "mu_bounds": self.ctrl.cfg.mu_bounds,
            "sigma_bounds": self.ctrl.cfg.sigma_bounds,
            "gains": self.ctrl.cfg.gains,
        }) + "\n")

    def on_evaluate(self, args, state, control, metrics, **kw):
        rec = {
            "type": "eval",
            "step": int(state.global_step),
            "epoch": float(state.epoch or 0.0),
            "time": time.time(),
            "eval_loss": float(metrics.get("eval_loss")) if "eval_loss" in metrics else None,
            "alpha": float(self.ctrl.alpha),
            "mu": float(self.ctrl.mu),
            "sigma": float(self.ctrl.sigma),
            # controller internals (for debugging theory)
            "el": self.ctrl.el, "er": self.ctrl.er, "vl": self.ctrl.vl,
            "Sdl": self.ctrl.Sdl, "Sdr": self.ctrl.Sdr, "Sv": self.ctrl.Sv,
            "dL_norm": getattr(self.ctrl, "_dL", None),
            "dR_norm": getattr(self.ctrl, "_dR", None),
        }
        self._f.write(json.dumps({k: v for k, v in rec.items() if v is not None}) + "\n")

        # Also log to Trainer so you can see curves live
        plateau = self._plateau_score_scratch()
        trainer = kw.get("trainer", None)
        if trainer is not None:
            trainer.log({
                "arl/alpha": self.ctrl.alpha,
                "arl/mu": self.ctrl.mu,
                "arl/sigma": self.ctrl.sigma,
                "arl/dL": getattr(self.ctrl, "_dL", float("nan")),
                "arl/dR": getattr(self.ctrl, "_dR", float("nan")),
                "arl/plateau": plateau,
                "eval/loss": float(metrics.get("eval_loss")) if "eval_loss" in metrics else float("nan"),
            })

    def _plateau_score_scratch(self):
        # Mirror the controller’s plateau logic for visibility
        try:
            dL = abs(getattr(self.ctrl, "_dL", 0.0))
            vl = float(self.ctrl.vl)
            Sv = float(self.ctrl.Sv) + 1e-8
            s = lambda z: 1.0 / (1.0 + math.exp(-z))
            a = s((self.ctrl.cfg.plateau_eps_d - dL) / 0.25)
            b = s((self.ctrl.cfg.plateau_eps_v - (vl / Sv)) / 0.25)
            return a * b
        except Exception:
            return float("nan")

    def on_train_end(self, args, state, control, **kw):
        try:
            self._f.flush(); self._f.close()
        except Exception:
            pass
