#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Reusable helpers for Teuken-style SFT jobs (or any HF/DeepSpeed run).

Nothing in here depends on a concrete model, dataset, or output folder;
pass those things in as arguments so the same functions work everywhere.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import wandb
from accelerate import Accelerator
from transformers import TrainerCallback
import math
from statistics import median
from typing import List, Optional

import torch
from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)


# ──────────────────────────── Callbacks ──────────────────────────────
class GradNormWandBCallback(TrainerCallback):
    """Logs total grad-norm to Weights & Biases each optimisation step."""

    def on_train_begin(self, *_, **__):
        self.last_grad_norm = None

    def _compute_grad_norm(self, trainer, model) -> torch.Tensor:
        if trainer.deepspeed:
            return trainer.deepspeed.get_grad_norm()

        grads = [p.grad.norm(2) for p in model.parameters() if p.grad is not None]
        if not grads:                               # list is empty right after resume / early eval
            return torch.tensor(0.0, device=next(model.parameters()).device)
        return torch.norm(torch.stack(grads))

    def on_backward_end(self, args, state, control, **kw):
        trainer, model = kw["trainer"], kw["model"]
        g = self._compute_grad_norm(trainer, model)
        g = trainer.accelerator.gather_for_metrics(g).mean()
        self.last_grad_norm = g.item()

        if trainer.accelerator.is_local_main_process:
            trainer.log({"grad_norm": self.last_grad_norm})

    def on_log(self, args, state, control, logs=None, **kw):
        if logs is not None and self.last_grad_norm is not None:
            logs["grad_norm"] = self.last_grad_norm


# ─────────────────────────── File helpers ────────────────────────────
def resolve_ckpt(ckpt_var: str | None, workdir: Path) -> Optional[str]:
    """
    Turn ``CKPT_PATH`` env-style variable into an absolute path (or ``None``).

    * ``None`` / ``"None"`` → start from scratch
    * ``"LAST"`` (case-ins.) → newest ``checkpoint-*`` under *workdir*
    * otherwise              → treated as literal path
    """
    if not ckpt_var or str(ckpt_var).lower() == "none":
        return None

    if str(ckpt_var).upper() == "LAST":
        ckpts = sorted(workdir.glob("checkpoint-*"),
                       key=lambda p: int(p.name.split("-")[-1]))
        return str(ckpts[-1]) if ckpts else None

    p = Path(ckpt_var).expanduser()
    if not p.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {p}")
    return str(p)


def save_ds_config(cfg: Dict[str, Any], file_path: str | Path) -> str:
    """Serialise a DeepSpeed JSON config and return its path."""
    file_path = Path(file_path)
    file_path.write_text(json.dumps(cfg, indent=2))
    return str(file_path)

# ────────────────────── Weights & Biases utils ───────────────────────
def init_or_resume_wandb(
        accelerator: Accelerator,
        run_id_file: Path,
        trainer_cfg,
        resume_ckpt: Optional[str],
        notes: str = "",
        project: str = "teuken-runs",
) -> None:
    """
    Make *resuming* completely transparent: the same run ID is reused
    if a checkpoint exists and we saw this run before.
    """
    if not accelerator.is_local_main_process:
        return

    # (1) figure out run-ID
    if resume_ckpt and run_id_file.exists():
        run_id = run_id_file.read_text().strip()
    else:
        run_id = wandb.util.generate_id()
        run_id_file.parent.mkdir(parents=True, exist_ok=True)
        run_id_file.write_text(run_id)

    # (2) start / resume the run
    wandb.init(
        project=project,
        name=trainer_cfg.run_name,
        id=run_id,
        resume="allow",
        notes=notes,
        config=(
            trainer_cfg.to_dict()
            if hasattr(trainer_cfg, "to_dict")
            else {k: v for k, v in vars(trainer_cfg).items()
                  if not k.startswith("_")}
        ),
    )


# ─────────────────────── Saving & Hub upload ─────────────────────────
def consolidate_and_save(
        accelerator: Accelerator,
        engine,
        tokenizer,
        target_dir: Path,
) -> None:
    """
    Gather ZeRO-3 partitions, unwrap the model, and store a normal HF checkpoint.
    Returns only on rank-0.
    """
    state_dict = accelerator.get_state_dict(engine)
    unwrapped  = accelerator.unwrap_model(engine)

    if accelerator.is_local_main_process:
        target_dir.mkdir(parents=True, exist_ok=True)
        unwrapped.save_pretrained(
            target_dir,
            is_main_process=True,
            save_function=accelerator.save,
            state_dict=state_dict,
            safe_serialization=True,
        )
        tokenizer.save_pretrained(target_dir)

        first_shard = next(target_dir.glob("model-*-of-*.safetensors"))
        shardsize_gb = first_shard.stat().st_size / 2**30
        print(f"✅ consolidated checkpoint (~{shardsize_gb:.1f} GB/shard) → {target_dir}")

    accelerator.wait_for_everyone()


def push_to_hf_hub(
        full_dir: Path,
        repo_id: str,
        hf_token: str,
        commit_message: str = "Upload fine-tuned checkpoint",
        private: bool = False,
) -> None:
    """Create (or reuse) the Hub repo and upload a folder."""
    from huggingface_hub import HfApi

    api = HfApi(token=hf_token)
    api.create_repo(repo_id=repo_id, repo_type="model",
                    private=private, exist_ok=True)

    api.upload_folder(
        repo_id=repo_id,
        folder_path=str(full_dir),
        path_in_repo=".",
        commit_message=commit_message,
    )
    print(f"🚀 pushed checkpoint to https://huggingface.co/{repo_id}")

# Perplexity helpers ---------------------------------------------------
def exp_from_loss(loss: float | torch.Tensor) -> float:
    """`math.exp(loss)` with a single call site so you never forget it."""
    if isinstance(loss, torch.Tensor):
        loss = loss.item()
    return math.exp(loss)


def shift_logits_and_labels(
    logits: torch.Tensor,
    labels: torch.Tensor,
):
    """Shift tokens so that <t> predicts token t (standard LM training)."""
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    return shift_logits, shift_labels


def compute_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Token-level cross-entropy independent of any `Trainer`."""
    shift_logits, shift_labels = shift_logits_and_labels(logits, labels)
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
    return loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    )


def build_ppl_compute_metrics(ignore_index: int = -100):
    """
    Factory that returns a `compute_metrics` fn compatible with HF Trainer.
    Usage in the training script:
        compute_metrics = build_ppl_compute_metrics()
    """
    import math  # local import avoids polluting utils' public namespace

    def _compute_metrics(eval_pred):
        logits, labels = eval_pred
        logits = torch.tensor(logits)
        labels = torch.tensor(labels)
        loss = compute_cross_entropy(logits, labels, ignore_index)
        return {"ppl": math.exp(loss.item())}

    return _compute_metrics

class EarlyStopBadRun(TrainerCallback):
    r"""
    **Early-termination heuristic from  
    “Unveiling the Secret Recipe: A Guide for Supervised Fine-Tuning Small LLMs”**.

    *Idea* (see § 3.7 of the paper):  
    ───────────────────────────────────  
    Good runs show **lower gradient-norms** *and* **higher training-losses** in
    the first ≈ 750 optimiser steps; bad runs show the opposite.  
    If both signals point to a bad run, abort to save compute.

    Parameters
    ----------
    window_steps : int
        Number of optimisation steps to observe before making a decision.
        Default = 750 (paper value, ≈ first 1-2 % of training).
    grad_norm_threshold : float
        Upper bound on the *median* gradient-norm considered healthy.
        Paper used ≈ 4.5 for 7 B models with 4 k–8 k effective batch.
    loss_threshold : float
        Lower bound on the *median* training loss considered healthy.
        < 0.90 (after tokenisation) was a reliable sign of over-fitting in the study.
    """

    def __init__(
        self,
        window_steps: int = 750,
        grad_norm_threshold: float = 4.5,
        loss_threshold: float = 0.90,
    ) -> None:
        self.window_steps = window_steps
        self.grad_norm_threshold = grad_norm_threshold
        self.loss_threshold = loss_threshold

        self._grad_hist: List[float] = []
        self._loss_hist: List[float] = []
        self._decision_made: bool = False

    # ───────────────────────────── callback ────────────────────────────
    def on_log(  # called every `logging_steps`
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Optional[dict] = None,
        **kwargs,
    ) -> TrainerControl:
        if logs is None or self._decision_made:
            return control

        # ── pick up values logged by other callbacks / Trainer ─────────
        grad_norm = logs.get("grad_norm")
        loss_val = logs.get("loss")

        # If grad_norm not provided try to compute it (last resort)
        if grad_norm is None and "model" in kwargs:
            total_norm = torch.norm(
                torch.stack(
                    [
                        p.grad.detach().norm(2)
                        for p in kwargs["model"].parameters()
                        if p.grad is not None
                    ]
                ),
                2,
            ).item()
            grad_norm = total_norm

        if grad_norm is not None:
            self._grad_hist.append(float(grad_norm))
        if loss_val is not None:
            self._loss_hist.append(float(loss_val))

        # ── After observation window, decide once ──────────────────────
        if state.global_step >= self.window_steps and not self._decision_made:
            self._decision_made = True  # ensure single evaluation

            med_grad = median(self._grad_hist) if self._grad_hist else float("inf")
            med_loss = median(self._loss_hist) if self._loss_hist else 0.0

            bad_grad = med_grad > self.grad_norm_threshold
            bad_loss = med_loss < self.loss_threshold

            if bad_grad and bad_loss:
                # mark training for early stop; no further checkpoints
                control.should_training_stop = True
                control.should_save = False
                if args.local_rank in (-1, 0):
                    print(
                        f"[EarlyStopBadRun] Stopping run at step {state.global_step} – "
                        f"median grad_norm={med_grad:.2f} (> {self.grad_norm_threshold}), "
                        f"median loss={med_loss:.3f} (< {self.loss_threshold})"
                    )

        return control
