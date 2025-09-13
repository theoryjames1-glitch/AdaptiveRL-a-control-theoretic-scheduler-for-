# ARLBandit.py
# K-armed bandit that switches AdaptiveRLStepController "modes" every window.
# Uses discounted Gaussian Thompson Sampling on a continuous reward.
# No sigils, No recursions, No glyphs, No rituals.

from dataclasses import dataclass
from typing import List, Dict, Optional
import math, time, copy
from transformers import TrainerCallback

# -------- Bandit core (Gaussian Thompson with discount) --------

@dataclass
class ArmStats:
    mean: float = 0.0    # posterior mean
    prec: float = 1e-6   # posterior precision (1/var)
    pulls: int = 0

class KArmedBanditTS:
    def __init__(self, n_arms: int, prior_mean=0.0, prior_var=1.0, obs_var=1.0, discount=0.9):
        self.n = n_arms
        self.obs_prec = 1.0 / max(obs_var, 1e-8)
        self.discount = float(discount)
        self.arms: List[ArmStats] = [ArmStats(prior_mean, 1.0/max(prior_var,1e-8), 0) for _ in range(n_arms)]
        self.t = 0

    def select(self):
        # Thompson: sample from N(mean, var) where var = 1/prec
        import random
        samples = []
        for a in self.arms:
            var = 1.0 / max(a.prec, 1e-12)
            # simple normal sample
            z = random.gauss(0.0, 1.0)
            samples.append(a.mean + (var ** 0.5) * z)
        idx = int(max(range(self.n), key=lambda i: samples[i]))
        return idx

    def update(self, idx: int, reward: float):
        self.t += 1
        # discount all arms to handle non-stationarity
        for a in self.arms:
            a.prec *= self.discount
        a = self.arms[idx]
        # posterior precision & mean update for Gaussian with known noise var
        a.prec = a.prec + self.obs_prec
        a.mean = (a.mean * (a.prec - self.obs_prec) + reward * self.obs_prec) / max(a.prec, 1e-12)
        a.pulls += 1

# -------- Bandit callback that reconfigures the controller --------

@dataclass
class BanditMode:
    name: str
    gains: Dict[str, float]                 # partial gains to override in controller.cfg.gains
    lr_change_cap: Optional[tuple] = None   # e.g., (0.9, 1.2)
    update_every: Optional[int] = None      # controller update cadence

class ARLBanditWindow(TrainerCallback):
    """
    Every `window_steps`, compute reward from loss improvement, update bandit,
    choose a mode, and apply it to the AdaptiveRLStepController cfg for the next window.
    Reward (default): normalized drop in mean loss over the window, clipped [-1, 1].
    """
    def __init__(self, controller, modes: List[BanditMode], window_steps=200,
                 prior_mean=0.0, prior_var=1.0, obs_var=0.25, discount=0.9,
                 trainer=None, echo=True):
        assert len(modes) >= 2, "Need at least 2 modes for a bandit."
        self.ctrl = controller
        self.modes = modes
        self.window_steps = int(window_steps)
        self.bandit = KArmedBanditTS(len(modes), prior_mean, prior_var, obs_var, discount)
        self._trainer = trainer
        self._echo = echo

        # window stats
        self._k = 0
        self._sum_loss = 0.0
        self._n_loss = 0
        self._last_window_mean = None
        self._current_mode = 0  # start with mode 0 by default
        self._apply_mode(0)

    def _get_tr(self, kw):
        return kw.get("trainer", self._trainer)

    def _apply_mode(self, idx: int):
        mode = self.modes[idx]
        # shallow copy gains and override only keys present in the mode
        g = copy.copy(self.ctrl.cfg.gains)
        g.update(mode.gains or {})
        self.ctrl.cfg.gains = g
        if mode.lr_change_cap is not None:
            self.ctrl.cfg.lr_change_cap = tuple(mode.lr_change_cap)
        if mode.update_every is not None:
            self.ctrl.cfg.update_every = int(mode.update_every)
        self._current_mode = idx
        if self._echo:
            print(f"[Bandit] -> mode='{mode.name}'  gains={mode.gains}  "
                  f"cap={self.ctrl.cfg.lr_change_cap}  upd_every={self.ctrl.cfg.update_every}")

    def on_log(self, args, state, control, logs=None, **kw):
        # accumulate loss inside window
        if logs and "loss" in logs:
            try:
                self._sum_loss += float(logs["loss"])
                self._n_loss += 1
            except Exception:
                pass

        self._k += 1
        if self._k % self.window_steps != 0:
            return

        # compute window reward
        cur_mean = (self._sum_loss / max(self._n_loss, 1)) if self._n_loss else None
        reward = 0.0
        if (cur_mean is not None) and (self._last_window_mean is not None):
            # positive if loss decreased
            raw = (self._last_window_mean - cur_mean)
            scale = abs(self._last_window_mean) + 1e-8
            reward = max(min(raw / scale, 1.0), -1.0)

        # update bandit
        self.bandit.update(self._current_mode, reward)
        next_mode = self.bandit.select()

        if self._echo:
            name = self.modes[self._current_mode].name
            print(f"[Bandit] window_end step={state.global_step}  last_mode='{name}'  "
                  f"cur_mean={cur_mean:.6g}  prev_mean={self._last_window_mean}  reward={reward:+.4f}")

        # reset window stats
        self._last_window_mean = cur_mean
        self._sum_loss = 0.0
        self._n_loss = 0

        # apply chosen mode for next window
        self._apply_mode(next_mode)

    # Optional: use eval loss at epoch ends as a stronger reward signal
    def on_evaluate(self, args, state, control, metrics, **kw):
        if "eval_loss" not in metrics:
            return
        ev = float(metrics["eval_loss"])
        reward = 0.0
        if getattr(self, "_last_eval", None) is not None:
            raw = (self._last_eval - ev)
            scale = abs(self._last_eval) + 1e-6
            reward = max(min(raw / scale, 1.0), -1.0)
            self.bandit.update(self._current_mode, reward)
            if self._echo:
                name = self.modes[self._current_mode].name
                print(f"[Bandit] eval_reward epoch={state.epoch} mode='{name}' "
                      f"prev_eval={self._last_eval:.6g} cur_eval={ev:.6g} reward={reward:+.4f}")
        self._last_eval = ev
