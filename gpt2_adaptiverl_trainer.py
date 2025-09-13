# gpt2_adaptiverl_trainer.py
# GPT-2 fine-tune with stepwise AdaptiveRL + per-step real LR/beta1 logging
# No sigils, No recursions, No glyphs, No rituals

import math, dataclasses
from packaging import version
from datasets import load_dataset
import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    TrainerCallback,
)

# Grad-noise injection + save/load shim (from your AdaptiveRL.py)
from AdaptiveRL import AdaptiveRLTrainer, ARLStateIO

# Stepwise controller & callback (from AdaptiveRLStep.py)
from AdaptiveRLStep import ARLStepConfig, AdaptiveRLStepController, AdaptiveRLStepCallback

# Per-step diagnostics with config (from ARLStepDiagnostics.py)
from ARLStepDiagnostics import ARLStepDiagnostics, ARLStepDiagConfig

HF_VER = version.parse(transformers.__version__)
FIELDS = {f.name for f in dataclasses.fields(TrainingArguments)}

# -------------------------
# 1) Data & Tokenization
# -------------------------
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

block_size = 1024
ds = load_dataset("wikitext", "wikitext-2-raw-v1")

def tokenize_fn(examples):
    return tokenizer(examples["text"])

def group_texts(examples):
    concatenated = {k: sum(examples[k], []) for k in examples}
    total_length = (len(concatenated["input_ids"]) // block_size) * block_size
    return {
        k: [t[i:i+block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated.items()
    }

tokenized = ds.map(tokenize_fn, batched=True, remove_columns=ds["train"].column_names)
packed = tokenized.map(group_texts, batched=True)

collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# -------------------------
# 2) Model
# -------------------------
model = AutoModelForCausalLM.from_pretrained(model_name)
model.config.pad_token_id = tokenizer.pad_token_id

# -------------------------
# 3) TrainingArguments (per-step logging; constant scheduler)
# -------------------------
args_kwargs = dict(
    output_dir="out-gpt2-arl-step",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=1,   # 1:1 step-to-log for clarity
    num_train_epochs=3,
    logging_steps=1,                 # log EVERY step
    learning_rate=8e-5,              # initial α; controller updates optimizer LR
    weight_decay=0.01,
    report_to=["none"],
)

# precision
if "bf16" in FIELDS:
    args_kwargs["bf16"] = True
elif "fp16" in FIELDS:
    args_kwargs["fp16"] = True

# step logging strategy (if supported)
if "logging_strategy" in FIELDS:
    args_kwargs["logging_strategy"] = "steps"

# keep scheduler flat so controller’s LR is visible
if "lr_scheduler_type" in FIELDS:
    args_kwargs["lr_scheduler_type"] = "constant"

if "remove_unused_columns" in FIELDS:
    args_kwargs["remove_unused_columns"] = False

# optional periodic evals just for monitoring
HAVE_EVAL_STRATEGY = "evaluation_strategy" in FIELDS
if HAVE_EVAL_STRATEGY:
    args_kwargs["evaluation_strategy"] = "steps"
    args_kwargs["eval_steps"] = 200
if "save_strategy" in FIELDS:
    args_kwargs["save_strategy"] = "no"

training_args = TrainingArguments(**args_kwargs)

# -------------------------
# 4) Stepwise AdaptiveRL controller (demo-visible settings)
# -------------------------
cfg = ARLStepConfig()
cfg.update_every = 1  # update EVERY logged step to make changes obvious
cfg.gains.update(
    kr=0.08, kl=0.08,           # LR reaction
    kmur=0.04, kmul=0.06,       # momentum (beta1) nudges
    ksigP=0.10, ksigR=0.08
)
cfg.lr_change_cap = (0.9, 1.2)   # up to ±10–20% per update
cfg.alpha_bounds = (1e-5, 2e-4)  # give room to move

step_ctrl = AdaptiveRLStepController(
    init_alpha=training_args.learning_rate,
    init_mu=0.9,
    init_sigma=0.0,
    cfg=cfg
)

# -------------------------
# 5) Trainer + Callbacks (stepwise control + real LR logging)
# -------------------------
trainer = AdaptiveRLTrainer(
    model=model,
    args=training_args,
    data_collator=collator,
    train_dataset=packed["train"],
    eval_dataset=packed["validation"],
)

# Stepwise control: α->optimizer.lr (preserving group ratios), μ->beta1, σ->grad-noise
trainer.add_callback(AdaptiveRLStepCallback(step_ctrl, trainer=trainer))  # <-- removed echo_updates

# Save/load controller with checkpoints
trainer.add_callback(ARLStateIO(step_ctrl))

# Per-step diagnostics: include actual opt LR/β1 and print every step
diag_cfg = ARLStepDiagConfig(
    echo_every=1,
    echo_columns=["step", "loss", "sched_lr", "opt_lr", "alpha", "mu", "sigma", "beta1_now"],
)
trainer.add_callback(ARLStepDiagnostics(step_ctrl, out_dir=training_args.output_dir, cfg=diag_cfg, trainer=trainer))

from ARLBandit import ARLBanditWindow, BanditMode

modes = [
    BanditMode(
        name="aggressive",
        gains=dict(kr=0.10, kl=0.10, kmur=0.05, kmul=0.07, ksigP=0.08, ksigR=0.08),
        lr_change_cap=(0.9, 1.25),
        update_every=1,
    ),
    BanditMode(
        name="steady",
        gains=dict(kr=0.05, kl=0.05, kmur=0.02, kmul=0.04, ksigP=0.06, ksigR=0.06),
        lr_change_cap=(0.9, 1.15),
        update_every=10,
    ),
    BanditMode(
        name="explore",
        gains=dict(kr=0.04, kl=0.04, kmur=0.02, kmul=0.03, ksigP=0.15, ksigR=0.05),
        lr_change_cap=(0.9, 1.20),
        update_every=5,
    ),
    BanditMode(
        name="cooling",
        gains=dict(kr=0.02, kl=0.08, kmur=0.01, kmul=0.05, ksigP=0.05, ksigR=0.08),
        lr_change_cap=(0.95, 1.05),
        update_every=5,
    ),
]

# Pick a window size (steps) for the bandit to evaluate & switch modes
trainer.add_callback(ARLBanditWindow(
    controller=step_ctrl,
    modes=modes,
    window_steps=200,          # try 100–400; smaller = faster reactions
    prior_mean=0.0,
    prior_var=0.25,
    obs_var=0.25,
    discount=0.9,
    trainer=trainer,
    echo=True,
))

# Fallback for very old HF (ensure periodic evals for monitoring only)
class _ForceEvalEveryNSteps(TrainerCallback):
    def __init__(self, n=200): self.n = int(n)
    def on_step_end(self, args, state, control, **kw):
        if state.global_step and state.global_step % self.n == 0:
            control.should_evaluate = True
        return control

if not HAVE_EVAL_STRATEGY:
    trainer.add_callback(_ForceEvalEveryNSteps(n=200))

# -------------------------
# 6) Train & Evaluate
# -------------------------
trainer.train()

metrics = trainer.evaluate()
ppl = math.exp(metrics["eval_loss"]) if "eval_loss" in metrics else float("nan")
print({"eval_loss": metrics.get("eval_loss", None), "perplexity": ppl})
