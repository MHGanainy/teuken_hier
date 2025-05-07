#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PyTorch-DDP / HF-Accelerate script for SFT-training Teuken-7B.

Examples
--------
Single node, 8 GPUs, bfloat16:
    CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --multi_gpu --num_processes 4 --mixed_precision bf16 src/train_ddp.py
"""
from __future__ import annotations

import os
from pathlib import Path

import torch
from dotenv import load_dotenv
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer
from accelerate import DistributedDataParallelKwargs

# ─────────────────────────── utils  ────────────────────────────
from utils import (
    GradNormWandBCallback,
    resolve_ckpt,
    init_or_resume_wandb,
    push_to_hf_hub,
    build_ppl_compute_metrics,
)

# ───────────────────────── credentials ─────────────────────────
load_dotenv()                                   # reads .env → os.environ
HF_TOKEN = os.getenv("HF_API_KEY")              # MUST exist
if HF_TOKEN is None:
    raise RuntimeError("HF_API_KEY not found in environment (.env)")

HUB_REPO_ID = "MHGanainy/teuken-ddp-hier-summ-sft"

# ────────────────────────── paths & constants ──────────────────────────
# MODEL_NAME = "openGPT-X/Teuken-7B-instruct-research-v0.4"
MODEL_NAME = "utter-project/EuroLLM-1.7B-Instruct"
DATA_DIR   = Path("/home/heshmo/workspace/teuken_hier/data/processed")
OUT_DIR    = Path("teuken-ddp-hier-sft")
FULL_DIR   = OUT_DIR / "best"              # final consolidated folder
RUN_ID_FILE = OUT_DIR / "wandb_run_id.txt"

# ─────────────────── checkpoint selection ─────────────────────────────
CKPT_PATH   = "LAST"                          # None / "LAST" / "/path/to/ckpt"
resume_ckpt = resolve_ckpt(CKPT_PATH, OUT_DIR)

# ───────────────────────── dataset ────────────────────────────────────
train_ds = (
    load_dataset("json", data_files=str(DATA_DIR / "train.jsonl"), split="train")
    .select([0])                       # tiny demo, single record
)
val_ds = train_ds

# ───────────────────── tokenizer & base model ─────────────────────────
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME, use_fast=False, trust_remote_code=True
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.bfloat16, trust_remote_code=True
)
model.gradient_checkpointing_enable()
model.config.dropout = 0.1

# ───────────────────────── trainer config ─────────────────────────────
compute_metrics = build_ppl_compute_metrics()

trainer_cfg = SFTConfig(
    output_dir=str(OUT_DIR),
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,   # → effective batch = world_size × 8
    num_train_epochs=22,
    learning_rate=2e-5,
    lr_scheduler_type="constant",
    warmup_steps=0,
    adam_beta1=0.9,
    adam_beta2=0.95,
    weight_decay=0.01,
    max_length=4096,
    packing=False,
    max_grad_norm = 1.0,
    logging_steps=1,
    completion_only_loss=True,
    eval_strategy="epoch",
    save_strategy="epoch",
    per_device_eval_batch_size=1,
    dataset_text_field="prompt",
    bf16=True,
    optim="adamw_bnb_8bit",
    eval_accumulation_steps=1,
    run_name="teuken-ddp-hier-sft",
    report_to=["wandb"],
    ddp_find_unused_parameters=False,
)

# ─────────────────────── trainer ──────────────────────────────────────
trainer = SFTTrainer(
    model=model,
    args=trainer_cfg,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[GradNormWandBCallback()],
)

# ─────────────────────── main routine ─────────────────────────────────
if __name__ == "__main__":

    # 0️⃣  WandB (rank-0 only)
    init_or_resume_wandb(
        accelerator=trainer.accelerator,
        run_id_file=RUN_ID_FILE,
        trainer_cfg=trainer_cfg,
        resume_ckpt=resume_ckpt,
        notes=(
            "Teuken-7B SFT — "
            f"{trainer.accelerator.num_processes}-GPU DDP run"
        ),
        project=os.getenv("WANDB_PROJECT", "teuken-hier"),
    )

    # 1️⃣  Train / resume
    trainer.train(resume_from_checkpoint=resume_ckpt)

    # 2️⃣  Consolidate & save a vanilla HF checkpoint
    if trainer.accelerator.is_local_main_process:
        trainer.save_model(FULL_DIR)
        tokenizer.save_pretrained(FULL_DIR)

    # 4️⃣  Push to the Hub (rank-0 only)
    if trainer.accelerator.is_local_main_process:
        push_to_hf_hub(
            full_dir=FULL_DIR,
            repo_id=HUB_REPO_ID,
            hf_token=HF_TOKEN,
        )

        import wandb
        wandb.finish()
