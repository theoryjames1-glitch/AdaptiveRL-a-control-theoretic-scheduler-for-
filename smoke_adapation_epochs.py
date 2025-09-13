# smoke_adaptation_epochs.py
# Causal AdaptiveRL epoch updates (works on older HF via control.should_evaluate)
# Run: python smoke_adaptation_epochs.py

import math, dataclasses
from packaging import version
import transformers
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    DataCollatorForLanguageModeling, TrainingArguments, TrainerCallback,
)
from AdaptiveRL import (
    AdaptiveRLEpochController, AdaptiveRLCallback, AdaptiveRLTrainer, ARLEpochConfig
)

FIELDS = {f.name for f in dataclasses.fields(TrainingArguments)}

def tiny_ds():
    lines = [
        "the quick brown fox jumps over the lazy dog.",
        "time flies like an arrow; fruit flies like a banana.",
        "gpt2 adaptive rl tiny corpus.",
        "simple line for language modeling.",
        "another short sentence to learn from.",
    ] * 200
    return Dataset.from_dict({"text": lines})

def tokenize_pack(tok, ds, block_size=128):
    tokd = ds.map(lambda ex: tok(ex["text"]), batched=True, remove_columns=["text"])
    def pack(ex):
        cat = {k: sum(ex[k], []) for k in ex}
        T = (len(cat["input_ids"]) // block_size) * block_size
        return {k: [t[i:i+block_size] for i in range(0, T, block_size)] for k, t in cat.items()}
    return tokd.map(pack, batched=True)

class ForceEvalEachEpoch(TrainerCallback):
    """Set the control flag so Trainer evaluates at each epoch end."""
    def on_epoch_end(self, args, state, control, **kw):
        control.should_evaluate = True
        return control

class PrintARL(TrainerCallback):
    """Print α/μ/σ after ARL updates on each evaluation."""
    def __init__(self, ctrl): self.ctrl = ctrl
    def on_evaluate(self, args, state, control, metrics, **kw):
        el = metrics.get("eval_loss")
        print(f"[ARL] epoch={float(state.epoch or 0):.1f}  eval_loss={None if el is None else round(el,4)}  "
              f"alpha={self.ctrl.alpha:.6g}  mu={self.ctrl.mu:.4f}  sigma={self.ctrl.sigma:.3g}")

if __name__ == "__main__":
    model_name = "gpt2"  # or "sshleifer/tiny-gpt2"
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.pad_token = tok.pad_token or tok.eos_token

    ds = tiny_ds()
    packed = tokenize_pack(tok, ds)
    coll = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.config.pad_token_id = tok.pad_token_id

    # Multi-epoch; we let ForceEvalEachEpoch trigger evals at epoch ends
    args_kwargs = dict(
        output_dir="out-smoke-adapt",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=1,
        num_train_epochs=4,         # multiple epochs so updates take effect
        logging_steps=50,
        learning_rate=5e-5,        # initial α
        weight_decay=0.0,
        report_to=["none"],
    )
    if "remove_unused_columns" in FIELDS: args_kwargs["remove_unused_columns"] = False
    if "bf16" in FIELDS: args_kwargs["bf16"] = True
    elif "fp16" in FIELDS: args_kwargs["fp16"] = True
    if "lr_scheduler_type" in FIELDS: args_kwargs["lr_scheduler_type"] = "constant"
    if "save_strategy" in FIELDS: args_kwargs["save_strategy"] = "no"

    training_args = TrainingArguments(**args_kwargs)

    # Use slightly stronger gains + wider LR cap so changes are visible
    cfg = ARLEpochConfig()
    cfg.gains.update(kr=0.12, kl=0.12, kmur=0.05, kmul=0.08, ksigP=0.10, ksigR=0.08)
    cfg.lr_change_cap = (0.8, 1.3)

    ctrl = AdaptiveRLEpochController(init_alpha=training_args.learning_rate, init_mu=0.9, init_sigma=0.0, cfg=cfg)

    trainer = AdaptiveRLTrainer(
        model=model,
        args=training_args,
        data_collator=coll,
        train_dataset=packed,
        eval_dataset=packed.select(range(min(128, len(packed)))),  # small eval slice is fine
    )

    # Add ARL callback (pass trainer for older HF), then print + force-eval callbacks
    trainer.add_callback(AdaptiveRLCallback(ctrl, trainer=trainer))
    trainer.add_callback(PrintARL(ctrl))
    trainer.add_callback(ForceEvalEachEpoch())

    print("[DEMO] starting train() …")
    trainer.train()
    print("[DEMO] training done. Final evaluate() …")
    m = trainer.evaluate()
    ppl = math.exp(m["eval_loss"]) if "eval_loss" in m else float("nan")
    print({"final_eval_loss": m.get("eval_loss"), "final_perplexity": ppl})
