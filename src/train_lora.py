#!/usr/bin/env python
"""
QLoRA SFT for Teuken‑7B‑instruct‑research‑v0.4 (arg‑free version)
================================================================
This script fine‑tunes Teuken‑7B with **QLoRA** (4‑bit NF4 quantisation + LoRA)
without relying on any command‑line flags. All tunable values live as constants
at the top of the file—edit them directly if you need different paths or
hyper‑parameters.

Tested with:
* Python ≥ 3.10, PyTorch ≥ 2.2, CUDA 11.8
* `pip install -U torch transformers[torch] trl peft datasets accelerate bitsandbytes einops`

Run it:
    python qlora_teuken_finetune.py
"""
from __future__ import annotations
from pathlib import Path
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from trl import SFTTrainer, SFTConfig

###############################################################################
# 1 – Fixed configuration values
###############################################################################
MODEL_NAME   = "openGPT-X/Teuken-7B-instruct-research-v0.4"
DATA_DIR     = Path("/home/heshmo/workspace/teuken_hier/data/processed")  # ← adjust as needed
OUTPUT_DIR   = Path("teuken-hier-qlora")

# LoRA / training hyper‑params
SEED             = 42
BF16             = True          # compute in bf16 (Teuken is bf16 native)
NUM_EPOCHS       = 150
BATCH_SIZE       = 1             # per‑device micro‑batch
GRAD_ACC_STEPS   = 1            # effective batch = BATCH_SIZE × GRAD_ACC_STEPS
LEARNING_RATE    = 2e-5
LORA_R           = 32
LORA_ALPHA       = 16
LORA_DROPOUT     = 0.05


###############################################################################
# 3 – Main
###############################################################################

def main():
    torch.manual_seed(SEED)

    # ── 3.1 Dataset ────────────────────────────────────────────────────────
    train_ds = load_dataset("json", data_files=str(DATA_DIR/"train.jsonl"), split="train").select([0])
    val_ds = train_ds
    # val_ds   = load_dataset("json", data_files=str(DATA_DIR/"val.jsonl"),   split="train")


    # ── 3.2 Tokenizer ─────────────────────────────────────────────────────
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False, trust_remote_code=True)

    # ── 3.3 Quantised base model ──────────────────────────────────────────
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if BF16 else torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    print("Loading quantised base model… (this can take a while)")
    mdl = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",          # let Accelerate pick GPUs
        trust_remote_code=True,
        quantization_config=bnb_cfg,
    )

    # prepare for 4‑bit training (adds input_casting + output_embedding fix)
    mdl = prepare_model_for_kbit_training(mdl)

    # ── 3.4 LoRA configuration ───────────────────────────────────────────
    lora_cfg = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )

    mdl = get_peft_model(mdl, lora_cfg)
    mdl.print_trainable_parameters()

    # ── 3.5 SFT (Supervised Fine‑Tuning) configuration ───────────────────
    sft_cfg = SFTConfig(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACC_STEPS,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="constant",
        warmup_steps=0,
        adam_beta1=0.9,
        adam_beta2=0.95,
        weight_decay=0.0,
        max_length=4096,
        packing=True,
        logging_steps=1,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        completion_only_loss=True,
        dataset_text_field="prompt",
        bf16=BF16,
        seed=SEED,
    )

    trainer = SFTTrainer(
        model=mdl,
        args=sft_cfg,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tok,
    )

    # ── 3.6 Training ─────────────────────────────────────────────────────
    trainer.train()

    # ── 3.7 Save & optional merge ────────────────────────────────────────
    print("Saving adapter weights…")
    trainer.model.save_pretrained(OUTPUT_DIR)
    tok.save_pretrained(OUTPUT_DIR)

    # (Optional) merge LoRA into one full model to avoid needing PEFT at serve time
    if torch.cuda.device_count() == 1:  # merging on multi‑GPU is slow / memory intense
        print("Merging adapters into base weights (fp16)…")
        merged_dir = OUTPUT_DIR / "merged"
        merged = mdl.merge_and_unload()
        merged.save_pretrained(merged_dir)
        tok.save_pretrained(merged_dir)

    # ── 3.8 Quick generation test ────────────────────────────────────────
    print("\n===== QUICK INFERENCE TEST =====")
    sample = train_ds[0]
    prompt_text = sample["prompt"]
    completion_text = sample["completion"]
    print("\n--- PROMPT ---\n" + prompt_text)
    print("\n--- COMPLETION ---\n" + completion_text)
    comp_token_len = len(tok(completion_text, add_special_tokens=False)["input_ids"]) if completion_text else 900
    print(f"*****Completion token length: {comp_token_len}")
    mdl.eval()
    with torch.no_grad():
        model_inputs = tok(prompt_text, return_tensors="pt").to(mdl.device)
        prompt_len = model_inputs["input_ids"].shape[-1]
        model_inputs.pop("token_type_ids", None)
        generated = mdl.generate(
            **model_inputs,
            max_new_tokens=comp_token_len,
            do_sample=True,
            top_p=0.9,
            temperature=0.1,
        )
    new_tokens = generated[0][prompt_len:]
    cleaned_text = tok.decode(new_tokens, skip_special_tokens=True).lstrip()
    print("\n--- MODEL OUTPUT ---\n" + cleaned_text)


###############################################################################
# Entry
###############################################################################
if __name__ == "__main__":
    main()
