# AdaptiveRLStep.py
# Stepwise AdaptiveRL controller + HF Trainer callback
# No sigils, No recursions, No glyphs, No rituals

from dataclasses import dataclass
import math
from transformers import TrainerCallback

@dataclass
class ARLStepConfig:
    alpha_bounds = (1e-5, 1.5e-4)
    mu_bounds    = (0.85, 0.99)   # maps to AdamW beta1
    sigma_bounds = (0.0, 0.1)
    gains = dict(kr=0.02, kl=0.02, kmur=0.01, kmul=0.02, ksigP=0.05, ksigR=0.04)
    ema_betas = dict(loss=0.02, reward=0.02, var=0.01, scale=0.02)
    update_every = 20             # <-- update cadence (steps)
    lr_change_cap = (0.8, 1.2)    # per-update multiplicative clamp for α
    plateau_eps_d = 0.05
    plateau_eps_v = 0.02

class AdaptiveRLStepController:
    """
    Step-level controller: observes loss/reward every log step, updates α/μ/σ every K steps.
    """
    def __init__(self, init_alpha, init_mu=0.9, init_sigma=0.0, cfg: ARLStepConfig = ARLStepConfig()):
        self.cfg = cfg
        self.alpha = float(init_alpha)
        self.mu    = float(init_mu)
        self.sigma = float(init_sigma)

        # EMAs & scales
        self.el = None; self.er = None; self.vl = 0.0
        self.Sdl = 1e-8; self.Sdr = 1e-8; self.Sv = 1e-8
        self._dL = 0.0; self._dR = 0.0
        self._prev_alpha = self.alpha
        self._eps = 1e-8
        self._k = 0  # step counter for cadence

    @staticmethod
    def _ema(old, new, beta): return new if old is None else (1 - beta) * old + beta * new
    @staticmethod
    def _sigm(z): return 1 / (1 + math.exp(-z))
    @staticmethod
    def _logit(p): return math.log(p / (1 - p))

    def observe_step(self, loss_value, reward_value=None):
        """Update EMAs and normalized deltas from a scalar loss (and optional reward)."""
        L = float(loss_value)
        R = float(-loss_value if reward_value is None else reward_value)

        bL = self.cfg.ema_betas["loss"]; bR = self.cfg.ema_betas["reward"]
        bV = self.cfg.ema_betas["var"];  bS = self.cfg.ema_betas["scale"]

        prev_el, prev_er = self.el, self.er
        self.el = self._ema(self.el, L, bL)
        self.er = self._ema(self.er, R, bR)

        # variance proxy around loss EMA
        self.vl = (1 - bV) * self.vl + bV * (L - (self.el if self.el is not None else L)) ** 2

        dL = 0.0 if prev_el is None else (self.el - prev_el)
        dR = 0.0 if prev_er is None else (self.er - prev_er)

        self.Sdl = (1 - bS) * self.Sdl + bS * (abs(dL) + self._eps)
        self.Sdr = (1 - bS) * self.Sdr + bS * (abs(dR) + self._eps)
        self.Sv  = (1 - bS) * self.Sv  + bS * (abs(self.vl) + self._eps)

        self._dL = max(min(dL / (self.Sdl + self._eps), 1.0), -1.0)
        self._dR = max(min(dR / (self.Sdr + self._eps), 1.0), -1.0)

    def _plateau_score(self):
        s = self._sigm
        a = s((self.cfg.plateau_eps_d - abs(self._dL)) / 0.25)                 # flat trend
        b = s((self.cfg.plateau_eps_v - (self.vl / (self.Sv + 1e-8))) / 0.25)  # low variance
        return a * b

    def maybe_update(self):
        """Every update_every steps, adjust α/μ/σ and return the new values; else None."""
        self._k += 1
        if self._k % self.cfg.update_every != 0:
            return None

        kr, kl   = self.cfg.gains["kr"],   self.cfg.gains["kl"]
        kmur, kmul = self.cfg.gains["kmur"], self.cfg.gains["kmul"]
        ksigP, ksigR = self.cfg.gains["ksigP"], self.cfg.gains["ksigR"]

        # α in log-space + per-update multiplicative cap
        loga = math.log(max(self.alpha, 1e-16)) + kr * self._dR - kl * max(self._dL, 0.0)
        new_alpha = math.exp(loga)
        lo, hi = self.cfg.lr_change_cap
        factor = min(max(new_alpha / max(self._prev_alpha, 1e-16), lo), hi)
        self.alpha = float(min(max(self._prev_alpha * factor, self.cfg.alpha_bounds[0]), self.cfg.alpha_bounds[1]))
        self._prev_alpha = self.alpha

        # μ via logit
        z = self._logit(min(max(self.mu, 1e-6), 1 - 1e-6)) + kmur * self._dR - kmul * max(self._dL, 0.0)
        self.mu = float(min(max(self._sigm(z), self.cfg.mu_bounds[0]), self.cfg.mu_bounds[1]))

        # σ in log-space
        logs = math.log(max(self.sigma, 1e-16)) + ksigP * self._plateau_score() - ksigR * max(self._dR, 0.0)
        self.sigma = float(min(max(math.exp(logs), self.cfg.sigma_bounds[0]), self.cfg.sigma_bounds[1]))

        return dict(alpha=self.alpha, mu=self.mu, sigma=self.sigma)

    # ---- resume safety (for ARLStateIO) ----
    def state_dict(self):
        return dict(
            alpha=self.alpha, mu=self.mu, sigma=self.sigma,
            el=self.el, er=self.er, vl=self.vl,
            Sdl=self.Sdl, Sdr=self.Sdr, Sv=self.Sv,
            _prev_alpha=self._prev_alpha, _k=self._k
        )

    def load_state_dict(self, sd: dict):
        self.alpha = float(sd["alpha"]); self.mu = float(sd["mu"]); self.sigma = float(sd["sigma"])
        self.el = sd["el"]; self.er = sd["er"]; self.vl = sd["vl"]
        self.Sdl = sd["Sdl"]; self.Sdr = sd["Sdr"]; self.Sv = sd["Sv"]
        self._prev_alpha = float(sd["_prev_alpha"]); self._k = int(sd.get("_k", 0))

class AdaptiveRLStepCallback(TrainerCallback):
    """
    Hooks into HF Trainer logs (every `logging_steps`) to:
      1) observe step loss
      2) maybe update α/μ/σ
      3) push new values to optimizer (preserving per-group LR ratios)
    """
    def __init__(self, controller: AdaptiveRLStepController, trainer=None):
        self.ctrl = controller
        self._trainer = trainer
        self._alpha_scales = None

    def _get_trainer(self, kwargs):
        return kwargs.get("trainer", self._trainer)

    def _compute_scales(self, opt):
        base = opt.param_groups[0]["lr"]
        return [pg["lr"] / max(base, 1e-12) for pg in opt.param_groups]

    def on_train_begin(self, args, state, control, **kwargs):
        tr = self._get_trainer(kwargs)
        if tr is not None:
            self._alpha_scales = self._compute_scales(tr.optimizer)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or "loss" not in logs:
            return
        self.ctrl.observe_step(loss_value=float(logs["loss"]))
        hp = self.ctrl.maybe_update()
        if hp is None:
            return

        tr = self._get_trainer(kwargs)
        if tr is None:
            return
        opt = tr.optimizer

        if self._alpha_scales is None or len(self._alpha_scales) != len(opt.param_groups):
            self._alpha_scales = self._compute_scales(opt)

        for pg, scale in zip(opt.param_groups, self._alpha_scales):
            pg["lr"] = float(hp["alpha"] * scale)
            if "betas" in pg:
                b1, b2 = pg["betas"]
                pg["betas"] = (float(hp["mu"]), b2)

        # hand new sigma to the (noise-injecting) Trainer subclass
        setattr(tr, "_arl_sigma", float(hp["sigma"]))

__all__ = ["ARLStepConfig", "AdaptiveRLStepController", "AdaptiveRLStepCallback"]
