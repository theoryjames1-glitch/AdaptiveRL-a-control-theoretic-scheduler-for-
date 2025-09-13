# AdaptiveRL: a control-theoretic scheduler for α, μ, σ

## 0) Objects and goals

* Optimizer parameters (plant input): learning rate **α** > 0, momentum **μ** ∈ (0,1) (β₁ for AdamW or momentum for SGD), exploration scale **σ** ≥ 0 (gradient noise).
* Signals we can read (plant output): per-step (or per-epoch) loss ℓₜ, optional task reward rₜ (e.g., −ℓₜ for LMs, return for RL).
* Objective: adapt (**α, μ, σ**) online to **speed convergence** and **escape plateaus** while preserving **stability** (no blow-ups) and **scale invariance** (don’t care about absolute ℓ/r units).

---

## 1) Estimators (the “sensors”)

Slow EMAs keep the controller on a slower time-scale than weight updates:

* Loss trend:

  $$
  \bar\ell_t=(1-\beta_\ell)\bar\ell_{t-1}+\beta_\ell \,\ell_t,\quad
  \Delta \ell_t=\bar\ell_t-\bar\ell_{t-1}
  $$
* Reward trend (optional):

  $$
  \bar r_t=(1-\beta_r)\bar r_{t-1}+\beta_r\, r_t,\quad
  \Delta r_t=\bar r_t-\bar r_{t-1}
  $$
* Loss variance proxy:

  $$
  v_t=(1-\beta_v) v_{t-1}+\beta_v(\ell_t-\bar\ell_t)^2
  $$
* Safe scales (unitless normalization): EMAs of absolute changes

  $$
  S_\ell\gets \text{EMA}(|\Delta \ell_t|),\quad S_r\gets \text{EMA}(|\Delta r_t|)
  $$

Normalized trends (clipped):

$$
\tilde{\Delta}\ell_t=\mathrm{clip}\!\left(\frac{\Delta \ell_t}{S_\ell+\epsilon},-1,1\right),\quad
\tilde{\Delta} r_t=\mathrm{clip}\!\left(\frac{\Delta r_t}{S_r+\epsilon},-1,1\right)
$$

**Plateau score** (0…1; high = flat & low-variance):

$$
P_t=\sigma_{\text{sig}}\!\left(\frac{\epsilon_\Delta-|\tilde{\Delta}\ell_t|}{s_\Delta}\right)\cdot
      \sigma_{\text{sig}}\!\left(\frac{\epsilon_v- v_t/(S_v+\epsilon)}{s_v}\right)
$$

with small slope scales $s_\Delta,s_v$. (Use $S_v=\text{EMA}(|v_t|)$ for normalization.)

**Interpretation**

* $\tilde{\Delta} r_t>0$: getting better → take bigger steps, more inertia, less noise.
* $\tilde{\Delta}\ell_t>0$: getting worse → cool LR, reduce inertia, maybe add noise.
* $P_t$ high: stuck/flat → increase exploration.

---

## 2) Update laws (the “controller”)

Work in spaces that enforce bounds by construction:

* α, σ in **log-space** (multiplicative updates, stay positive),
* μ in **logit-space** (stays in (0,1)).

Let $k_\bullet$ be small gains.

**Learning rate**

$$
\log \alpha_{t+1}=\log \alpha_t + k_r\,\tilde{\Delta} r_t - k_\ell \max(\tilde{\Delta}\ell_t,0)
$$

then clamp $\alpha_{t+1}\in[\alpha_{\min},\alpha_{\max}]$ and optionally cap per-update factor:
$\alpha_{t+1}\in[\alpha_t\!\cdot\!c_\downarrow,\;\alpha_t\!\cdot\!c_\uparrow]$.

**Momentum (β₁)**

$$
\mathrm{logit}(\mu_{t+1})=\mathrm{logit}(\mu_t)+k_{\mu r}\,\tilde{\Delta} r_t-k_{\mu \ell}\max(\tilde{\Delta}\ell_t,0)
$$

then $\mu_{t+1}=\sigma_{\text{sig}}(\cdot)$ and clamp to $[\mu_{\min},\mu_{\max}]$.

**Exploration (gradient noise)**

$$
\log \sigma_{t+1}=\log \sigma_t+k_{\sigma P} P_t-k_{\sigma r}\max(\tilde{\Delta} r_t,0)
$$

then clamp $\sigma_{t+1}\in[\sigma_{\min},\sigma_{\max}]$.

**Noise injection (if σ>0)**
Prefer gradient noise for stability:

$$
g_t^{\text{noisy}}=g_t+\underbrace{\big(\sigma_t \cdot \mathrm{RMS}(g_t)+\epsilon\big)}_{\sigma_t^{\text{eff}}}\xi,\quad \xi\sim\mathcal N(0,I)
$$

(Per-tensor RMS scaling is fine; parameter noise is a riskier alternative.)

**Two time-scales**

* Update controller every $K$ steps (**stepwise**) or once per epoch (**epoch-wise**).
* Use small gains $k\in[10^{-3},10^{-1}]$.
* EMA betas small (e.g., 0.01–0.05 stepwise; larger if epoch-wise).

---

## 3) Safety & invariants (why it won’t blow up)

* **Boundedness (trivial invariants).** Clipping in log/logit space ensures
  $\alpha\in[\alpha_{\min},\alpha_{\max}],\;\mu\in[\mu_{\min},\mu_{\max}],\;\sigma\in[\sigma_{\min},\sigma_{\max}]$ for all t.
* **Scale-invariance.** All decisions use normalized $\tilde{\Delta}\ell,\tilde{\Delta} r$ and normalized variance $v/S_v$; absolute magnitudes don’t matter.
* **Two-time-scale stability (SA/ODE view).** With small gains and $K\ge 10$, the controller evolves slower than the optimizer. Under Lipschitz/bounded-noise assumptions (standard in stochastic approximation), the closed-loop tracks a stable ODE of the optimizer with quasi-static $(\alpha,\mu,\sigma)$. Practically: set gains small, verify no LR oscillations; reduce gains or increase $K$ if you see “hunting”.
* **Monotone sign-reflex.** By construction, $\tilde{\Delta} r>0$ pushes $\alpha$ up and $\sigma$ down; $\tilde{\Delta}\ell>0$ pushes $\alpha$ and $\mu$ down. This protects against runaway LR when things worsen.

---

## 4) Integration patterns with optimizers/schedulers

* **AdamW:** map μ → β₁; leave β₂ fixed (0.99–0.999) initially.
* **SGD(m):** set momentum directly.
* **Scheduler relationship:**

  1. **Replace** (simplest): use constant scheduler; controller writes param-group LRs.
  2. **Modulate** (optional): keep a cosine/warmup baseline $\text{lr}_\text{sched}(t)$, let controller be a multiplicative factor $\alpha_t\in[\alpha_\min,\alpha_\max]$ around 1.0, so $\text{lr}_\text{eff}(t)=\text{lr}_\text{sched}(t)\cdot\alpha_t$.

---

## 5) Stepwise vs epoch-wise (causal)

* **Stepwise controller** (updates every $K$ steps): fastest reflex, works without evaluations; recommended when you can log loss every step.
* **Epoch-wise controller**: simpler wiring with HF `Trainer` (update at `on_evaluate`), slower reflex; good for tasks where reward/val is only available at epoch boundaries.

---

## 6) Defaults (strong but safe)

* Bounds:
  $\alpha_{\min}=10^{-5},\ \alpha_{\max}\in[1\!\times\!10^{-4},2\!\times\!10^{-4}]$ (GPT-2-small)
  $\mu_{\min}=0.85,\ \mu_{\max}=0.99$
  $\sigma_{\min}=0,\ \sigma_{\max}\in[0.05,0.2]$ (0 for LMs unless plateauing)
* Gains (stepwise demo):
  $k_r=k_\ell\in[0.02,0.08],\ k_{\mu r}\in[0.01,0.05],\ k_{\mu \ell}\in[0.02,0.06],\ k_{\sigma P}\in[0.05,0.10],\ k_{\sigma r}\in[0.04,0.08]$
* Update cadence: $K=10$–20 (stepwise) or per epoch.
* EMA betas: stepwise 0.01–0.03; epoch-wise 0.2–0.4.
* LR change cap: $(c_\downarrow,c_\uparrow)=(0.9,1.2)$ (demo), tighten to (0.95,1.05) for production smoothness.

---

## 7) What the controller should do (qualitative predictions)

* Early training (improvement): $\alpha \uparrow, \mu \uparrow, \sigma \downarrow\rightarrow 0$.
* Plateau: $P\uparrow \Rightarrow \sigma \uparrow$ (gentle exploration); $\alpha$ holds or cools.
* Regression (loss trend ↑): $\alpha \downarrow, \mu \downarrow$ (less inertia), $\sigma$ may increase briefly to search, then decay as improvement resumes.
* Distribution shift: brief dip in $\alpha,\mu$ + bump in $\sigma$, then recovery.

---

## 8) How to test the theory (causal only)

**Metrics (same step budget across methods):** final perplexity/return, AUC of −eval\_loss vs steps, time-to-target (e.g., PPL ≤ 30).
**Baselines:** constant LR, cosine, AdamW default, AdamW+cosine.
**Diagnostics:** plot $\alpha,\mu,\sigma$ with loss/reward; scatter $\Delta r$ vs Δlogα (should be positively correlated), $\max(\tilde{\Delta}\ell,0)$ vs −Δlogα (positive).
**Ablations:** loss-only vs reward-only, σ frozen vs adaptive, noise on grad vs param.

---

## 9) Extensions (optional future work)

* **Bandit meta-controller (k-armed):** choose among discrete “modes” (gain/cap/update\_every presets) every window; reward = normalized loss drop in window; use discounted Thompson Sampling to handle drift.
* **Non-causal (“hindsight”):** two-pass smoothing or hypergradient/adjoint to optimize past schedules offline, then **distill** into a causal controller $f_\phi(\text{past features})$.
* **Relative-to-scheduler mode:** treat $\alpha_t$ as a factor around 1.0 that multiplies a baseline schedule.
* **State augmentation:** add gradient RMS/variance, generalization pulse (short-horizon val change), or curvature proxies.

---

## 10) Minimal algorithm (stepwise)

**Inputs:** gains $k_\bullet$, bounds, EMAs $\beta_\bullet$, cadence $K$.
**Loop (every step):**

1. Compute loss ℓₜ (and reward rₜ if available).
2. Update EMAs $\bar\ell, \bar r, v, S_\ell, S_r, S_v$; form $\tilde{\Delta}\ell, \tilde{\Delta} r, P$.
3. If step % $K$ = 0:

   * Update $\log \alpha,\ \mathrm{logit}(\mu),\ \log \sigma$ with laws above.
   * Clip to bounds and apply LR change cap.
   * **Write** α→optimizer param-group LRs (preserving ratios), μ→β₁, set σ for gradient-noise injection.
4. (If σ>0) add gradient noise before optimizer.step().

