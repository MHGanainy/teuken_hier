#!/usr/bin/env python
# -*- coding: utf-8 -*-
# CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --multi_gpu --num_processes 4 --mixed_precision bf16 src/train_fsdp.py
from __future__ import annotations

import os
from pathlib import Path

import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

# ─────────────────────────── utils  ────────────────────────────
# All reusable helpers live in utils.py
from utils import (
    GradNormWandBCallback,
    resolve_ckpt,
    save_ds_config,
    init_or_resume_wandb,
    consolidate_and_save,
    push_to_hf_hub,
    build_ppl_compute_metrics,
    EarlyStopBadRun
)
from datasets import load_dataset
import random, numpy as np

# ───────────────────────── seed ─────────────────────────
seed = 42
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

# ───────────────────────── credentials ─────────────────────────
load_dotenv()                                   # reads .env → os.environ
HF_TOKEN = os.getenv("HF_API_KEY")              # MUST exist
if HF_TOKEN is None:
    raise RuntimeError("HF_API_KEY not found in environment (.env)")

HUB_REPO_ID = "MHGanainy/teuken-fsdp-hier-summ-sft"

# ────────────────────────── paths & constants ──────────────────────────
MODEL_NAME = "openGPT-X/Teuken-7B-instruct-research-v0.4"
DATA_DIR   = Path("/home/heshmo/workspace/teuken_hier/data/processed")
OUT_DIR    = Path("teuken-fsdp-hier-sft")
FULL_DIR   = OUT_DIR / "best"              # final consolidated folder
RUN_ID_FILE = OUT_DIR / "wandb_run_id.txt"

# ─────────────────── checkpoint selection ─────────────────────────────
CKPT_PATH   = None                          # None / "LAST" / "/path/to/ckpt"
resume_ckpt = resolve_ckpt(CKPT_PATH, OUT_DIR)

# ─────────────────── DeepSpeed ZeRO-3 runtime config ──────────────────
ds_cfg = {
    "train_micro_batch_size_per_gpu": 1,
    "gradient_accumulation_steps":   4096,
    "gradient_clipping":             1.0,
    "zero_optimization": {
        "stage": 3,
        "allgather_partitions": True,
        "reduce_scatter":       True,
        "overlap_comm":         True,
        "contiguous_gradients": True,
        "stage3_gather_16bit_weights_on_model_save": True,
        "offload_param":     {"device": "none"},
        "offload_optimizer": {"device": "none"},
    },
    "bf16": {"enabled": True},
}
DS_JSON = save_ds_config(ds_cfg, "ds_zero3.json")

# ───────────────────────── dataset ────────────────────────────────────
train_ds = (load_dataset("json", data_files=str(DATA_DIR / "train.jsonl"), split="train"))
val_ds = load_dataset("json", data_files=str(DATA_DIR/"test.jsonl"),   split="train")

# ───────────────────── tokenizer & base model ─────────────────────────
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME, use_fast=False, trust_remote_code=True
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.bfloat16, trust_remote_code=True, attn_implementation="flash_attention_2"
)
model.gradient_checkpointing_enable()
model.config.use_cache = False
model.config.dropout = 0.1 

compute_metrics = build_ppl_compute_metrics() 

# ───────────────────────── trainer config ─────────────────────────────
trainer_cfg = SFTConfig(
    seed=seed,
    output_dir=str(OUT_DIR),
    deepspeed=DS_JSON,
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
    run_name="teuken-fsdp-hier-sft",
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
    # compute_metrics=compute_metrics,
    callbacks=[GradNormWandBCallback(),EarlyStopBadRun(window_steps=750,
                        grad_norm_threshold=4.5,
                        loss_threshold=0.90)],
)

# ─────────────────────── main routine ─────────────────────────────────
if __name__ == "__main__":

    # ─── Weights-and-Biases initialisation (rank-0 only) ──────────────
    init_or_resume_wandb(
        accelerator=trainer.accelerator,
        run_id_file=RUN_ID_FILE,
        trainer_cfg=trainer_cfg,
        resume_ckpt=resume_ckpt,
        notes="Teuken-7B SFT with ZeRO-3",
        project=os.getenv("WANDB_PROJECT", "teuken-hier"),
    )

    # 1️⃣ training run (fresh or resumed)
    trainer.train(resume_from_checkpoint=resume_ckpt)

    # 2️⃣ obtain the still-wrapped DeepSpeed engine
    engine = trainer.deepspeed or trainer.model_wrapped

    # 3️⃣ consolidate & save a normal HF checkpoint
    consolidate_and_save(
        trainer.accelerator, engine, tokenizer, FULL_DIR
    )

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
