# smoke_AdaptiveRLTrainer.py
# Quick smoke test for AdaptiveRLTrainer wiring + callbacks
# - builds a tiny synthetic dataset
# - runs a short training with an evaluation
# - checks LR (alpha), beta1 (mu), and sigma are sane & within bounds
# - prints a PASS/FAIL summary

import os, math, dataclasses
from packaging import version
import transformers
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    TrainerCallback,
)
from AdaptiveRL import (
    AdaptiveRLEpochController,
    AdaptiveRLTrainer,
    AdaptiveRLCallback,
    ARLStateIO,
)

HF_VER = version.parse(transformers.__version__)
FIELDS = {f.name for f in dataclasses.fields(TrainingArguments)}

def make_tiny_text_dataset(n_lines=200, repeat=16):
    base = [
        "the quick brown fox jumps over the lazy dog.",
        "lorem ipsum dolor sit amet, consectetur adipiscing elit.",
        "gpt2 adaptive rl smoke test sample line.",
        "time flies like an arrow fruit flies like a banana.",
        "simple tiny corpus for rapid unit testing.",
    ]
    lines = (base * repeat)[:n_lines]
    return Dataset.from_dict({"text": lines})

def tokenize_and_pack(tokenizer, ds, block_size=128):
    tok = ds.map(lambda ex: tokenizer(ex["text"]), batched=True, remove_columns=["text"])
    def pack(ex):
        concat = {k: sum(ex[k], []) for k in ex}
        total = (len(concat["input_ids"]) // block_size) * block_size
        return {k: [t[i:i+block_size] for i in range(0, total, block_size)] for k, t in concat.items()}
    return tok.map(pack, batched=True)

class ForceEvalEveryNSteps(TrainerCallback):
    """Fallback for older HF that lack evaluation_strategy='steps'."""
    def __init__(self, n=50): self.n = n
    def on_step_end(self, args, state, control, **kw):
        tr = kw.get("trainer")
        if tr and state.global_step and state.global_step % self.n == 0:
            tr.evaluate()

if __name__ == "__main__":
    model_name = "gpt2"  # small & common; switch to "sshleifer/tiny-gpt2" if you prefer
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Tiny synthetic dataset
    raw = make_tiny_text_dataset(n_lines=200, repeat=16)
    packed = tokenize_and_pack(tok, raw, block_size=128)
    collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)

    # Model
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.config.pad_token_id = tok.pad_token_id

    # TrainingArguments (feature-detected for older HF)
    args_kwargs = dict(
        output_dir="out-smoke-arl",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=1,
        num_train_epochs=1,
        max_steps=120,                 # short run
        logging_steps=25,
        learning_rate=5e-5,
        weight_decay=0.0,
        report_to=["none"],
        remove_unused_columns=False,
    )
    # precision flags
    if "bf16" in FIELDS:
        args_kwargs["bf16"] = True
    elif "fp16" in FIELDS:
        args_kwargs["fp16"] = True

    # scheduler
    if "lr_scheduler_type" in FIELDS:
        args_kwargs["lr_scheduler_type"] = "constant"

    # evaluation & saving
    have_eval_strategy = "evaluation_strategy" in FIELDS
    have_save_strategy = "save_strategy" in FIELDS
    if have_eval_strategy:
        # Trigger eval mid-run to exercise the ARL callback
        args_kwargs["evaluation_strategy"] = "steps"
        args_kwargs["eval_steps"] = 60
    if have_save_strategy:
        args_kwargs["save_strategy"] = "no"

    training_args = TrainingArguments(**args_kwargs)

    # Controller & Trainer
    ctrl = AdaptiveRLEpochController(init_alpha=training_args.learning_rate, init_mu=0.9, init_sigma=0.0)

    trainer = AdaptiveRLTrainer(
        model=model,
        args=training_args,
        data_collator=collator,
        train_dataset=packed,      # use same tiny set for eval in this smoke test
        eval_dataset=packed.select(range(min(64, len(packed)))),
    )

    # Callbacks
    trainer.add_callback(AdaptiveRLCallback(ctrl, trainer=trainer))  # pass trainer for older HF
    trainer.add_callback(ARLStateIO(ctrl))
    if not have_eval_strategy:
        trainer.add_callback(ForceEvalEveryNSteps(n=60))

    # --- Run ---
    print("[SMOKE] starting train()…")
    trainer.train()
    print("[SMOKE] train() finished, running evaluate()…")
    metrics = trainer.evaluate()
    print("[SMOKE] eval metrics:", metrics)

    # --- Checks ---
    # Grab optimizer state
    opt = trainer.optimizer
    pg0 = opt.param_groups[0]
    lr_now = float(pg0["lr"])
    mu_now = float(pg0["betas"][0]) if "betas" in pg0 else None
    sigma_now = float(getattr(trainer, "_arl_sigma", 0.0) or 0.0)

    # Bounds from controller
    a_lo, a_hi = ctrl.cfg.alpha_bounds
    m_lo, m_hi = ctrl.cfg.mu_bounds
    s_lo, s_hi = ctrl.cfg.sigma_bounds

    ok = True
    if not (a_lo - 1e-12 <= lr_now <= a_hi + 1e-12):
        print(f"[SMOKE][WARN] lr={lr_now} outside bounds {ctrl.cfg.alpha_bounds}"); ok = False
    if mu_now is not None and not (m_lo - 1e-12 <= mu_now <= m_hi + 1e-12):
        print(f"[SMOKE][WARN] beta1(mu)={mu_now} outside bounds {ctrl.cfg.mu_bounds}"); ok = False
    if not (s_lo - 1e-12 <= sigma_now <= s_hi + 1e-12):
        print(f"[SMOKE][WARN] sigma={sigma_now} outside bounds {ctrl.cfg.sigma_bounds}"); ok = False
    if "eval_loss" not in metrics:
        print("[SMOKE][WARN] no eval_loss found; evaluation may not have run."); ok = False

    print(f"[SMOKE] lr={lr_now:.6g}, mu(beta1)={mu_now if mu_now is not None else 'N/A'}, sigma={sigma_now:.4g}")
    print("[SMOKE] PASS" if ok else "[SMOKE] FAIL")
