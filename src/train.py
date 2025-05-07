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
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer
from datasets import load_dataset

# ─────────────────────────── utils  ────────────────────────────
from utils import (
    GradNormWandBCallback,
    resolve_ckpt,
    init_or_resume_wandb,
    push_to_hf_hub,
    build_ppl_compute_metrics,
    EarlyStopBadRun,
)

# ───────────────────────── seed ─────────────────────────
seed = 42
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

# ───────────────────────── credentials ─────────────────────────
load_dotenv()
HF_TOKEN = os.getenv("HF_API_KEY")
if HF_TOKEN is None:
    raise RuntimeError("HF_API_KEY not found in environment (.env)")

HUB_REPO_ID = "MHGanainy/teuken-hier-summ-sft"

# ────────────────────────── paths & constants ──────────────────────────
MODEL_NAME = "openGPT-X/Teuken-7B-instruct-research-v0.4"
DATA_DIR   = Path("/home/heshmo/workspace/teuken_hier/data/processed")
OUT_DIR    = Path("teuken-hier-sft")
FULL_DIR   = OUT_DIR / "best"
RUN_ID_FILE = OUT_DIR / "wandb_run_id.txt"

# ─────────────────── checkpoint selection ─────────────────────────────
CKPT_PATH   = None                          # None / "LAST" / "/path/to/ckpt"
resume_ckpt = resolve_ckpt(CKPT_PATH, OUT_DIR)

# ───────────────────────── dataset ────────────────────────────────────
train_ds = load_dataset("json", data_files=str(DATA_DIR / "train.jsonl"), split="train")
val_ds   = load_dataset("json", data_files=str(DATA_DIR / "test.jsonl"),  split="train")

# ───────────────────── tokenizer & base model ─────────────────────────
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME, use_fast=False, trust_remote_code=True
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    attn_implementation="flash_attention_2",
)
model.gradient_checkpointing_enable()
model.config.use_cache = False
model.config.dropout   = 0.1

compute_metrics = build_ppl_compute_metrics()

# ───────────────────────── trainer config ─────────────────────────────
trainer_cfg = SFTConfig(
    seed=seed,
    output_dir=str(OUT_DIR),
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4096,
    num_train_epochs=4,
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
    eval_steps=2,
    save_strategy="steps",
    save_total_limit=5,
    load_best_model_at_end=True,
    save_steps=2,
    per_device_eval_batch_size=1,
    dataset_text_field="prompt",
    bf16=True,
    optim="adamw_bnb_8bit",
    eval_accumulation_steps=1,
    run_name="teuken-hier-sft",
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
        EarlyStopBadRun(
            window_steps=750,
            grad_norm_threshold=4.5,
            loss_threshold=0.90,
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
        notes="Teuken-7B SFT — single-GPU run",
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
