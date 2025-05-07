#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Quick single-GPU generation test for the consolidated checkpoint in
`teuken-hier-sft/full/`.
"""

from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

# ───────────────────────── paths ─────────────────────────
DATA_DIR   = Path("/home/heshmo/workspace/teuken_hier/data/processed")
CKPT_DIR   = "MHGanainy/teuken-ddp-hier-summ-sft"          # output of training script
DEVICE     = "cuda"
# MODEL_NAME = "openGPT-X/Teuken-7B-instruct-research-v0.4"
MODEL_NAME = "utter-project/EuroLLM-1.7B-Instruct"

# ─────────────────── load tokenizer & model ──────────────
print(f"Loading checkpoint from {CKPT_DIR} …")

tok  = AutoTokenizer.from_pretrained(
    MODEL_NAME,          # use the **fine-tuned** tokenizer
    use_fast=False,
    trust_remote_code=True,
)

# prefer bf16, fall back to fp16 if the GPU can’t do bf16
dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

model = AutoModelForCausalLM.from_pretrained(
    CKPT_DIR,
    torch_dtype=dtype,
    device_map="auto",           # single-GPU → everything on cuda:0
    trust_remote_code=True,
).eval()

# ───────────────────────── dataset ───────────────────────
train_ds = load_dataset(
    "json",
    data_files=str(DATA_DIR / "train.jsonl"),
    split="train",
).select([0])

# ───────────────────── quick inference ───────────────────
sample          = train_ds[0]
prompt_text     = sample["prompt"]
completion_text = sample.get("completion", "")

print("\n===== QUICK INFERENCE TEST =====")
print("\n--- PROMPT ---\n"      + prompt_text)
print("\n--- COMPLETION ---\n" + completion_text)

comp_tok_len = (
    len(tok(completion_text, add_special_tokens=False)["input_ids"])
    if completion_text
    else 900
)
print(f"⚙️  max_new_tokens set to: {comp_tok_len}")

with torch.no_grad():
    inputs   = tok(prompt_text, return_tensors="pt")
    prompt_len = inputs["input_ids"].shape[-1]

    # place tensors on same device as first weight shard
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    inputs.pop("token_type_ids", None)        # some Llama-like models add it

    generated = model.generate(
        **inputs,
        max_new_tokens=comp_tok_len,
        do_sample=True,
        top_p=0.9,
        temperature=0.1,
    )

new_tokens   = generated[0][prompt_len:]
decoded_text = tok.decode(new_tokens, skip_special_tokens=True).lstrip()

print("\n--- MODEL OUTPUT ---\n" + decoded_text)
