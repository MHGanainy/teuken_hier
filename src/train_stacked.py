#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Example launch on 4 GPUs:
# CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --multi_gpu --num_processes 4 --mixed_precision bf16 src/train.py
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer, EarlyStoppingCallback
from trl import SFTConfig, SFTTrainer
from datasets import load_dataset, interleave_datasets

# ─────────────────────────── utils  ────────────────────────────
from utils import (
    GradNormWandBCallback,
    resolve_ckpt,
    init_or_resume_wandb,
    push_to_hf_hub,
    build_ppl_compute_metrics,
)

# ───────────────────────── seed ─────────────────────────
seed = 42
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

# ───────────────────────── credentials ─────────────────────────
load_dotenv()
HF_TOKEN = os.getenv("HF_API_KEY") #"REMOVED" #os.getenv("HF_API_KEY")
if HF_TOKEN is None:
    raise RuntimeError("HF_API_KEY not found in environment (.env)")

HUB_REPO_ID = "MHGanainy/teuken-hier-summ-sft-press-stacked"

# ────────────────────────── paths & constants ──────────────────────────
MODEL_NAME = "openGPT-X/Teuken-7B-instruct-research-v0.4"
TOKENIZER_NAME = "openGPT-X/Teuken-7B-instruct-research-v0.4"
DATA_DIR   = Path("/teuken_hier/data/processed")
OUT_DIR    = Path("teuken-hier-sft-press-summary-stacked")
FULL_DIR   = OUT_DIR / "best"
RUN_ID_FILE = OUT_DIR / "wandb_run_id.txt"
# ─────────────────── checkpoint selection ─────────────────────────────
CKPT_PATH   = None                          # None / "LAST" / "/path/to/ckpt"
resume_ckpt = resolve_ckpt(CKPT_PATH, OUT_DIR)

# ───────────────────────── dataset ────────────────────────────────────
abs_train = load_dataset("json", data_files=str(DATA_DIR / "abs_train.jsonl"), split="train")
abs_val   = load_dataset("json", data_files=str(DATA_DIR / "abs_test.jsonl"),  split="train")

hier_train = load_dataset(
   "json", data_files=str(DATA_DIR / "hier_train.jsonl"), split="train"
)
hier_val = load_dataset(
   "json", data_files=str(DATA_DIR / "hier_test.jsonl"), split="train"
)

TEMPERATURE = 0.5

sizes   = np.array([len(hier_train), len(abs_train)], dtype=float)
probs   = (sizes / sizes.sum()) ** (1 / TEMPERATURE)
probs   = probs / probs.sum()
print(f"Sampling probabilities  hier:{probs[0]:.2%}  abs:{probs[1]:.2%}")

train_ds = interleave_datasets(
    [hier_train, abs_train],
    probabilities=list(probs),
    seed=seed,
    stopping_strategy="all_exhausted",
)

val_ds = abs_val

# ───────────────────── tokenizer & base model ─────────────────────────
tokenizer = AutoTokenizer.from_pretrained(
    TOKENIZER_NAME, use_fast=False, trust_remote_code=True
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    attn_implementation="flash_attention_2",
)
# model.gradient_checkpointing_enable()
model.config.use_cache = False
model.config.dropout   = 0.2

compute_metrics = build_ppl_compute_metrics()

# ───────────────────────── trainer config ─────────────────────────────
trainer_cfg = SFTConfig(
    seed=seed,
    output_dir=str(OUT_DIR),
    per_device_train_batch_size=1,
    gradient_accumulation_steps=512,
    num_train_epochs=10,
    learning_rate=2e-5,
    lr_scheduler_type="constant",
    warmup_steps=0,
    adam_beta1=0.9,
    adam_beta2=0.95,
    weight_decay=0.01,
    max_length=4096,
    packing=False,
    logging_steps=1,
    completion_only_loss=True,
    eval_strategy="steps",
    eval_steps=25,
    save_strategy="steps",
    save_total_limit=2,
    save_steps=25,
    per_device_eval_batch_size=1,
    dataset_text_field="prompt",
    bf16=True,
    optim="adamw_torch_fused",
    load_best_model_at_end = True,      
    metric_for_best_model  = "eval_loss",
    greater_is_better      = False,  
    eval_accumulation_steps=1,
    run_name="teuken-hier-sft-press-summary-stacked",
    report_to=["wandb"],
    remove_unused_columns=False,
    max_grad_norm=1.0,
)

trainer = SFTTrainer(
    model=model,
    args=trainer_cfg,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    processing_class=tokenizer,
    # compute_metrics=compute_metrics,  # Uncomment if you want PPL evaluation
    callbacks=[
        GradNormWandBCallback(),
        EarlyStoppingCallback(
        early_stopping_patience   = 2,   
        early_stopping_threshold  = 0.0
    ),
    ],
)

# ─────────────────────── main routine ─────────────────────────────────
if __name__ == "__main__":

    # ─── Weights-and-Biases initialisation (rank-0 only) ──────────────
    init_or_resume_wandb(
        accelerator=trainer.accelerator,
        run_id_file=RUN_ID_FILE,
        trainer_cfg=trainer_cfg,
        resume_ckpt=resume_ckpt,
        notes="Teuken-7B SFT — single-GPU run with press summary stacked",  
        project=os.getenv("WANDB_PROJECT", "teuken-hier"),
    )

    # 1️⃣ training run (fresh or resumed)
    trainer.train(resume_from_checkpoint=resume_ckpt)

    # 2️⃣ obtain the accelerate-wrapped model (no DS engine now)
    engine = trainer.model_wrapped

    # 3️⃣ save a normal HF checkpoint
    if trainer.accelerator.is_local_main_process:
        trainer.save_model(FULL_DIR)
        tokenizer.save_pretrained(FULL_DIR)

    # 4️⃣ push to the Hub (rank-0 only)
    if trainer.accelerator.is_local_main_process:
        push_to_hf_hub(
            full_dir=FULL_DIR,
            repo_id=HUB_REPO_ID,
            hf_token=HF_TOKEN,
        )

    # 5️⃣ finish WandB
    if trainer.accelerator.is_local_main_process:
        import wandb
        wandb.finish()
