# AdaptiveRL.py
# Plug-and-play epoch-level AdaptiveRL controller + HF Trainer wiring
# No sigils, No recursions, No glyphs, No rituals

# pip install transformers datasets accelerate
from dataclasses import dataclass
import math, os, glob
import torch
from transformers import Trainer, TrainerCallback


# ----------------------------
# 1) Epoch-level Controller
# ----------------------------

@dataclass
class ARLEpochConfig:
    # GPT-2 defaults (tune per model/scale)
    alpha_bounds = (1e-5, 1e-4)   # learning rate bounds
    mu_bounds    = (0.85, 0.99)   # maps to AdamW beta1 bounds
    sigma_bounds = (0.0, 0.1)     # gradient noise scale bounds
    gains = dict(kr=0.05, kl=0.05, kmur=0.02, kmul=0.05, ksigP=0.08, ksigR=0.06)
    ema_betas = dict(loss=0.3, reward=0.3, var=0.3, scale=0.3)  # epoch-scale EMAs
    lr_change_cap = (0.8, 1.2)    # per-epoch multiplicative LR clamp
    plateau_eps_d = 0.05          # "flat" tolerance (normalized |Δℓ|)
    plateau_eps_v = 0.02          # low-variance tolerance (normalized var)


class AdaptiveRLEpochController:
    """
    Epoch-level controller that adjusts:
      - alpha: global LR (preserving per-group ratios in the Trainer callback)
      - mu:    AdamW beta1
      - sigma: gradient noise scale (used by the Trainer subclass)
    """

    def __init__(self, init_alpha, init_mu=0.9, init_sigma=0.0, cfg=ARLEpochConfig()):
        self.cfg = cfg
        self.alpha = float(init_alpha)
        self.mu    = float(init_mu)
        self.sigma = float(init_sigma)

        # EMAs and scales
        self.el = None; self.er = None; self.vl = 0.0
        self.Sdl = 1e-8; self.Sdr = 1e-8; self.Sv = 1e-8
        self._prev_alpha = self.alpha
        self._eps = 1e-8

        # cached normalized deltas
        self._dL = 0.0
        self._dR = 0.0

    @staticmethod
    def _ema(old, new, beta): return new if old is None else (1 - beta) * old + beta * new
    @staticmethod
    def _sigm(z): return 1 / (1 + math.exp(-z))
    @staticmethod
    def _logit(p): return math.log(p / (1 - p))

    def observe_epoch(self, loss_eval, loss_train=None, reward=None):
        """
        Observe epoch-level signals.
        By default uses eval loss for both ℓ and reward proxy: r := -eval_loss.
        If you have a separate validation metric, pass it via `reward`.
        """
        L = float(loss_eval if loss_train is None else loss_eval)  # keep interface simple
        R = float((-loss_eval) if reward is None else reward)

        bL = self.cfg.ema_betas["loss"]; bR = self.cfg.ema_betas["reward"]
        bV = self.cfg.ema_betas["var"];  bS = self.cfg.ema_betas["scale"]

        prev_el, prev_er = self.el, self.er
        self.el = self._ema(self.el, L, bL)
        self.er = self._ema(self.er, R, bR)

        # simple variance proxy around eval loss EMA
        self.vl = (1 - bV) * self.vl + bV * (L - self.el) ** 2

        dL = 0.0 if prev_el is None else (self.el - prev_el)
        dR = 0.0 if prev_er is None else (self.er - prev_er)

        self.Sdl = (1 - bS) * self.Sdl + bS * (abs(dL) + self._eps)
        self.Sdr = (1 - bS) * self.Sdr + bS * (abs(dR) + self._eps)
        self.Sv  = (1 - bS) * self.Sv  + bS * (abs(self.vl) + self._eps)

        self._dL = max(min(dL / (self.Sdl + self._eps), 1.0), -1.0)
        self._dR = max(min(dR / (self.Sdr + self._eps), 1.0), -1.0)

    def _plateau_score(self):
        s = self._sigm
        a = s((self.cfg.plateau_eps_d - abs(self._dL)) / 0.25)             # flat trend
        b = s((self.cfg.plateau_eps_v - (self.vl / (self.Sv + 1e-8))) / 0.25)  # low variance
        return a * b

    def update_for_next_epoch(self):
        # α (log space) with per-epoch factor cap
        kr, kl = self.cfg.gains["kr"], self.cfg.gains["kl"]
        loga = math.log(max(self.alpha, 1e-16)) + kr * self._dR - kl * max(self._dL, 0.0)
        new_alpha = math.exp(loga)
        if self.cfg.lr_change_cap:
            lo, hi = self.cfg.lr_change_cap
            factor = min(max(new_alpha / max(self._prev_alpha, 1e-16), lo), hi)
            new_alpha = self._prev_alpha * factor
        self.alpha = float(min(max(new_alpha, self.cfg.alpha_bounds[0]), self.cfg.alpha_bounds[1]))
        self._prev_alpha = self.alpha

        # μ (logit space) -> AdamW beta1
        kmur, kmul = self.cfg.gains["kmur"], self.cfg.gains["kmul"]
        z = self._logit(min(max(self.mu, 1e-6), 1 - 1e-6)) + kmur * self._dR - kmul * max(self._dL, 0.0)
        self.mu = self._sigm(z)
        self.mu = float(min(max(self.mu, self.cfg.mu_bounds[0]), self.cfg.mu_bounds[1]))

        # σ (log space) — small for LMs; plateau raises, reward gain decays
        ksigP, ksigR = self.cfg.gains["ksigP"], self.cfg.gains["ksigR"]
        logs = math.log(max(self.sigma, 1e-16)) + ksigP * self._plateau_score() - ksigR * max(self._dR, 0.0)
        self.sigma = float(min(max(math.exp(logs), self.cfg.sigma_bounds[0]), self.cfg.sigma_bounds[1]))

        return dict(alpha=self.alpha, mu=self.mu, sigma=self.sigma)

    # ---- resume safety ----
    def state_dict(self):
        return dict(
            alpha=self.alpha, mu=self.mu, sigma=self.sigma,
            el=self.el, er=self.er, vl=self.vl,
            Sdl=self.Sdl, Sdr=self.Sdr, Sv=self.Sv,
            _prev_alpha=self._prev_alpha
        )

    def load_state_dict(self, sd: dict):
        self.alpha = float(sd["alpha"]); self.mu = float(sd["mu"]); self.sigma = float(sd["sigma"])
        self.el = sd["el"]; self.er = sd["er"]; self.vl = sd["vl"]
        self.Sdl = sd["Sdl"]; self.Sdr = sd["Sdr"]; self.Sv = sd["Sv"]
        self._prev_alpha = float(sd["_prev_alpha"])


# -----------------------------------
# 2) Trainer Callback (epoch updates)
# -----------------------------------

class AdaptiveRLCallback(TrainerCallback):
    """
    Updates AdamW (lr, beta1) at each evaluation (epoch) using the controller.
    Preserves per-group LR ratios measured at train start.

    Pass `trainer` at construction if your HF version doesn't provide it in callback kwargs.
    """
    def __init__(self, controller: AdaptiveRLEpochController, trainer: Trainer | None = None):
        self.ctrl = controller
        self._alpha_scales = None  # per-group LR ratios
        self._trainer = trainer    # optional direct reference (older HF versions)

    def _get_trainer(self, kwargs):
        # Prefer HF-provided kwarg; fallback to constructor-provided trainer
        return kwargs.get("trainer", self._trainer)

    def _compute_scales(self, opt):
        base_lr = opt.param_groups[0]["lr"]
        return [pg["lr"] / max(base_lr, 1e-12) for pg in opt.param_groups]

    def on_train_begin(self, args, state, control, **kwargs):
        trainer = self._get_trainer(kwargs)
        if trainer is None:
            # Older HF: we'll lazily compute scales on first on_evaluate
            return
        opt = trainer.optimizer
        self._alpha_scales = self._compute_scales(opt)

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        trainer = self._get_trainer(kwargs)
        if trainer is None:
            # If we still can't reach the trainer, we can't apply hyperparams safely.
            # Just observe trends and bail.
            if "eval_loss" in metrics:
                self.ctrl.observe_epoch(loss_eval=float(metrics["eval_loss"]))
                self.ctrl.update_for_next_epoch()
            return

        opt = trainer.optimizer

        # Safety: compute or refresh scales if needed
        if self._alpha_scales is None or len(self._alpha_scales) != len(opt.param_groups):
            self._alpha_scales = self._compute_scales(opt)

        if "eval_loss" not in metrics:
            return

        L = float(metrics["eval_loss"])
        self.ctrl.observe_epoch(loss_eval=L)
        hp = self.ctrl.update_for_next_epoch()

        # Apply α and μ (β1) while preserving param-group ratios
        for pg, scale in zip(opt.param_groups, self._alpha_scales):
            pg["lr"] = float(hp["alpha"] * scale)
            if "betas" in pg:  # AdamW-like param group
                b1, b2 = pg["betas"]
                pg["betas"] = (float(hp["mu"]), b2)

        # Expose current sigma to the trainer (for noise injection)
        setattr(trainer, "_arl_sigma", float(hp["sigma"]))


# ------------------------------------------------
# 3) Save/Load ARL state alongside HF checkpoints
# ------------------------------------------------

class ARLStateIO(TrainerCallback):
    """
    Saves controller state_dict to each checkpoint and reloads it on resume.
    """
    def __init__(self, controller: AdaptiveRLEpochController, fname: str = "arl_controller.pt"):
        self.ctrl = controller
        self.fname = fname

    def on_save(self, args, state, control, **kwargs):
        ck = f"checkpoint-{state.global_step}"
        path = os.path.join(args.output_dir, ck, self.fname)
        try:
            torch.save(self.ctrl.state_dict(), path)
        except Exception as e:
            print(f"[ARLStateIO] Warning: failed to save controller state to {path}: {e}")

    def on_load(self, args, state, **kwargs):
        # Load most recent checkpoint's ARL state if present
        try:
            ckpts = sorted(
                glob.glob(os.path.join(args.output_dir, "checkpoint-*")),
                key=lambda p: int(p.split("-")[-1])
            )
            if not ckpts:
                return
            path = os.path.join(ckpts[-1], self.fname)
            if os.path.exists(path):
                sd = torch.load(path, map_location="cpu")
                self.ctrl.load_state_dict(sd)
                print(f"[ARLStateIO] Loaded controller state from {path}")
        except Exception as e:
            print(f"[ARLStateIO] Warning: failed to load controller state: {e}")


# -------------------------------------------------------
# 4) DDP-consistent gradient noise + AMP-safe injection
# -------------------------------------------------------

def _ddp_shared_generator(step: int, device: torch.device) -> torch.Generator:
    """
    Returns a torch.Generator seeded identically across ranks (if DDP is initialized).
    Falls back to a per-process generator if DDP isn't active.
    """
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            seed_tensor = torch.tensor([int(step)], device=device, dtype=torch.int64)
            # Broadcast seed from rank 0 so all ranks share the same noise realization
            dist.broadcast(seed_tensor, src=0)
            g = torch.Generator(device=device)
            g.manual_seed(int(seed_tensor.item()))
            return g
    except Exception:
        pass
    g = torch.Generator(device=device)
    g.manual_seed(int(step))
    return g


class AdaptiveRLTrainer(Trainer):
    """
    Injects noise into gradients before optimizer.step() (when sigma > 0).
    Works with or without AMP (Accelerate). If using DDP, the noise is made
    rank-consistent by deriving a shared RNG seed from the global step.
    """

    def optimizer_step(self, *args, **kwargs):
        sigma = float(getattr(self, "_arl_sigma", 0.0) or 0.0)
        if sigma > 0:
            # Unscale grads if AMP is on
            try:
                self.accelerator.unscale_gradients(self.optimizer)
            except Exception:
                pass

            eps = 1e-8
            step = int(self.state.global_step)
            device = next(self.model.parameters()).device
            gen = _ddp_shared_generator(step, device)

            with torch.no_grad():
                for p in self.model.parameters():
                    if p.grad is None or not p.requires_grad:
                        continue
                    g = p.grad
                    # scale noise to grad RMS (per-parameter tensor)
                    scale = sigma * (g.detach().pow(2).mean().sqrt() + eps)
                    noise = torch.randn_like(g, generator=gen)
                    g.add_(noise * scale)

        return super().optimizer_step(*args, **kwargs)


__all__ = [
    "ARLEpochConfig",
    "AdaptiveRLEpochController",
    "AdaptiveRLCallback",
    "ARLStateIO",
    "AdaptiveRLTrainer",
]
