#!/usr/bin/env python
"""dataset_prep.py

Creates stacked training / validation JSON‑Lines files for hierarchical
summarisation fine‑tuning of *Teuken‑7B‑instruct‑research‑v0.4*.

*   Reads the source CSV located at
        /home/heshmo/workspace/teuken_hier/data/raw/raw.csv
    which must contain at least the columns: ``id``, ``level``, ``prompt``,
    and ``summary`` (additional columns are ignored).
*   Shuffles all rows so every abstraction level is mixed ("stacked").
*   Splits 90 % / 10 % into train / validation sets with a fixed seed so the
    split is reproducible.
*   Converts every row to a single *Teuken* chat‑template string using the
    function ``build_chat``.
*   Writes the resulting strings to JSONL files
        /home/heshmo/workspace/teuken_hier/data/processed/train.jsonl
        /home/heshmo/workspace/teuken_hier/data/processed/val.jsonl

Run:
    python dataset_prep.py [--seed 42]
"""
from __future__ import annotations

import argparse
import json
import pathlib
from typing import Dict, Any, Iterable

import pandas as pd
from transformers import AutoTokenizer
from tqdm.auto import tqdm

###############################################################################
# Configuration
###############################################################################
RAW_CSV = pathlib.Path("/home/heshmo/workspace/teuken_hier/data/raw/raw.csv")
OUTPUT_DIR = pathlib.Path("/home/heshmo/workspace/teuken_hier/data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_NAME = "openGPT-X/Teuken-7B-instruct-research-v0.4"

###############################################################################
# Helpers
###############################################################################

def build_sample(row: Dict[str, Any], tokenizer) -> Dict[str, str]:
    """Create a valid Teuken DE chat‑template string for one sample.

    The tokenizer's built‑in "DE" chat_template accepts only **User** and
    **Assistant** roles, which *must* alternate.  A separate System role would
    raise the ``Roles must alternate User/Assistant/...`` TemplateError you
    encountered.  Instead, we prepend the general system description to the
    first User message.
    """
    user_text = (
        f"{row['prompt']}"
    )

    messages = [
        {"role": "User", "content": user_text}
    ]

    prompt_str = tokenizer.apply_chat_template(
        messages,
        chat_template="DE",
        tokenize=False,
        add_generation_prompt=True,
    )

    return {"prompt": prompt_str, "completion": row["summary"]}


def write_jsonl(path: pathlib.Path, samples: Iterable[Dict[str, str]], total: int):
    """Write dicts as JSONL."""
    with path.open("w", encoding="utf-8") as fp:
        for sample in tqdm(samples, total=total, desc=path.name, unit=" rows"):
            json.dump(sample, fp, ensure_ascii=False)
            fp.write("\n")

###############################################################################
# Main
###############################################################################

def main(seed: int = 42):
    # Load + shuffle dataset
    df = pd.read_csv(RAW_CSV, dtype={"level": str})

    if "split_name" not in df.columns:
        raise ValueError("CSV must contain a `split_name` column with"
                         " values train / val / test")

    print("Unique split_name values:", df["split_name"].unique())

    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False, trust_remote_code=True)

    counts: Dict[str, int] = {}
    for split in ("train", "validation", "test"):
        df_split = df.loc[df["split_name"].str.lower() == split]
        counts[split] = len(df_split)

        samples = (build_sample(r, tok) for _, r in df_split.iterrows())
        write_jsonl(OUTPUT_DIR / f"{split}.jsonl", samples, total=counts[split])

    print("\nPrepared dataset →", OUTPUT_DIR)
    for k in ("train", "validation", "test"):
        print(f"  {k:<11}: {counts[k]} lines")



if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    main(seed=args.seed)
