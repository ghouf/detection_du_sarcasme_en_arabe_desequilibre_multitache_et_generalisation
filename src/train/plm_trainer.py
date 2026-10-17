"""Fine-tuning pipeline for Partie E: Mono-task and Multi-task PLMs.

Supported PLMs:
- aubmindlab/bert-base-arabertv02-twitter
- UBC-NLP/MARBERTv2
"""

from __future__ import annotations

import copy
import os
import random
import time
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, average_precision_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from models.multitask import MonoTask, MultiTask

# Label mappings
SENTIMENT_MAP = {"NEG": 0, "NEU": 1, "POS": 2}
SENTIMENT_REV_MAP = {v: k for k, v in SENTIMENT_MAP.items()}

DIALECT_MAP = {"egypt": 0, "gulf": 1, "levant": 2, "magreb": 3, "msa": 4}
DIALECT_REV_MAP = {v: k for k, v in DIALECT_MAP.items()}


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def compute_sarcasm_class_weights(train_labels: pd.Series | np.ndarray) -> torch.Tensor:
    """Compute balanced class weights on the training split only:

    w_c = N / (2 * N_c)
    """
    labels = np.asarray(train_labels).astype(int)
    counts = np.bincount(labels, minlength=2)
    total = len(labels)
    w0 = total / (2.0 * max(1, counts[0]))
    w1 = total / (2.0 * max(1, counts[1]))
    return torch.tensor([w0, w1], dtype=torch.float32)


def choose_decision_threshold(
    y_true: np.ndarray,
    proba: np.ndarray,
    grid: np.ndarray | None = None,
) -> tuple[float, float]:
    """Find the threshold tau in [0.01, 0.99] that maximizes the sarcastic F1 score.

    Must be evaluated strictly on the validation set.
    """
    if grid is None:
        grid = np.round(np.arange(0.01, 1.00, 0.01), 2)
    labels = np.asarray(y_true).astype(int)
    scores = np.asarray(proba, dtype=float)

    best_thresh = 0.5
    best_f1 = -1.0
    for t in grid:
        preds = (scores >= t).astype(int)
        score = float(f1_score(labels, preds, pos_label=1, zero_division=0))
        if score > best_f1:
            best_f1 = score
            best_thresh = float(t)
    return best_thresh, best_f1


class EarlyStopping:
    """Early stopping to prevent overfitting, monitoring validation sarcastic F1."""

    def __init__(self, patience: int = 2, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_f1 = -1.0
        self.best_threshold = 0.5
        self.counter = 0
        self.early_stop = False
        self.best_state_dict: dict[str, torch.Tensor] | None = None

    def step(self, val_f1: float, threshold: float, model: nn.Module) -> bool:
        """Update tracker. Returns True if this is a new best checkpoint."""
        if val_f1 > self.best_f1 + self.min_delta:
            self.best_f1 = val_f1
            self.best_threshold = threshold
            self.best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False

    def restore_best(self, model: nn.Module) -> None:
        """Reload the checkpoint that achieved the highest validation sarcastic F1."""
        if self.best_state_dict is not None:
            model.load_state_dict(self.best_state_dict)


class ArSarcasmDataset(Dataset):
    """PyTorch Dataset for ArSarcasm-v2 holding tweets and multitask labels."""

    def __init__(
        self,
        df: pd.DataFrame,
        text_column: str = "tweet",
        preprocess_fn=None,
    ):
        self.texts = []
        raw_texts = df[text_column].fillna("").astype(str).tolist()
        if preprocess_fn is not None:
            self.texts = [preprocess_fn(t) for t in raw_texts]
        else:
            self.texts = raw_texts

        # Sarcasm (0 or 1)
        self.sarcasm = df["sarcasm"].astype(int).tolist()

        # Sentiment (0: NEG, 1: NEU, 2: POS)
        if "sentiment" in df.columns:
            self.sentiment = [SENTIMENT_MAP.get(str(s).strip().upper(), 1) for s in df["sentiment"]]
        else:
            self.sentiment = [1] * len(df)

        # Dialect (0: egypt, 1: gulf, 2: levant, 3: magreb, 4: msa)
        if "dialect" in df.columns:
            self.dialect = [DIALECT_MAP.get(str(d).strip().lower(), 4) for d in df["dialect"]]
        else:
            self.dialect = [4] * len(df)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, object]:
        return {
            "text": self.texts[idx],
            "y_sarc": self.sarcasm[idx],
            "y_sent": self.sentiment[idx],
            "y_dial": self.dialect[idx],
        }


def make_collate_fn(tokenizer, max_length: int = 128):
    """Dynamic padding collator for batch efficiency."""

    def collate_fn(batch: list[dict[str, object]]) -> dict[str, torch.Tensor]:
        texts = [b["text"] for b in batch]
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded["y_sarc"] = torch.tensor([b["y_sarc"] for b in batch], dtype=torch.long)
        encoded["y_sent"] = torch.tensor([b["y_sent"] for b in batch], dtype=torch.long)
        encoded["y_dial"] = torch.tensor([b["y_dial"] for b in batch], dtype=torch.long)
        return encoded

    return collate_fn


@dataclass
class TrainConfig:
    model_name: str = "aubmindlab/bert-base-arabertv02-twitter"
    task_mode: Literal["mono", "multi"] = "multi"
    lr: float = 2e-5
    batch_size: int = 32
    max_epochs: int = 4
    patience: int = 2
    warmup_ratio: float = 0.10
    weight_decay: float = 0.01
    lam_sent: float = 0.5  # lambda for sentiment loss: {0.3, 0.5, 1.0}
    use_dialect: bool = False
    lam_dial: float = 0.2
    max_len: int = 128
    dropout: float = 0.2
    fp16: bool = True
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def get_optimizer_and_scheduler(model: nn.Module, config: TrainConfig, total_steps: int):
    """AdamW with weight decay exclusion on bias & LayerNorm, plus linear warmup."""
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": config.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=config.lr)
    warmup_steps = int(config.warmup_ratio * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    return optimizer, scheduler


def evaluate_model(
    model: nn.Module,
    val_loader: DataLoader,
    config: TrainConfig,
    threshold: float | None = None,
) -> dict[str, float]:
    """Evaluate model on validation split: computes sarcastic F1, macro-F1, PR-AUC, accuracy."""
    model.eval()
    all_sarc_probs = []
    all_sarc_trues = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(config.device)
            attention_mask = batch["attention_mask"].to(config.device)

            if config.task_mode == "mono":
                logits_sarc = model(input_ids, attention_mask)
            else:
                outputs = model(input_ids, attention_mask)
                logits_sarc = outputs[0]

            probs = torch.softmax(logits_sarc, dim=-1)[:, 1].cpu().numpy()
            all_sarc_probs.extend(probs)
            all_sarc_trues.extend(batch["y_sarc"].numpy())

    all_sarc_probs = np.asarray(all_sarc_probs)
    all_sarc_trues = np.asarray(all_sarc_trues)

    # If threshold not provided, find the optimal threshold on this split
    if threshold is None:
        best_thresh, _ = choose_decision_threshold(all_sarc_trues, all_sarc_probs)
    else:
        best_thresh = threshold

    preds = (all_sarc_probs >= best_thresh).astype(int)
    f1_sarc = float(f1_score(all_sarc_trues, preds, pos_label=1, zero_division=0))
    f1_mac = float(f1_score(all_sarc_trues, preds, average="macro", zero_division=0))
    pr_auc_score = float(average_precision_score(all_sarc_trues, all_sarc_probs))
    acc = float(accuracy_score(all_sarc_trues, preds))

    return {
        "val_f1_sarcasm": f1_sarc,
        "val_macro_f1": f1_mac,
        "val_pr_auc": pr_auc_score,
        "val_accuracy": acc,
        "threshold": best_thresh,
        "probs": all_sarc_probs,
        "trues": all_sarc_trues,
    }


def train_plm(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    config: TrainConfig,
    verbose: bool = True,
) -> tuple[nn.Module, dict[str, float]]:
    """Complete training loop with early stopping, mixed precision, and threshold tuning."""
    set_seed(config.seed)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    # Preprocessing hook if model is AraBERT
    preprocess_fn = None
    if "arabert" in config.model_name.lower():
        try:
            from arabert.preprocess import ArabertPreprocessor

            prep = ArabertPreprocessor(model_name=config.model_name.split("/")[-1])
            preprocess_fn = prep.preprocess
        except Exception:
            preprocess_fn = None

    # Datasets and DataLoaders
    train_ds = ArSarcasmDataset(train_df, preprocess_fn=preprocess_fn)
    val_ds = ArSarcasmDataset(val_df, preprocess_fn=preprocess_fn)
    collate_fn = make_collate_fn(tokenizer, max_length=config.max_len)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # Initialize model
    if config.task_mode == "mono":
        model = MonoTask(config.model_name, dropout=config.dropout).to(config.device)
    else:
        model = MultiTask(
            config.model_name,
            use_dialect=config.use_dialect,
            dropout=config.dropout,
        ).to(config.device)

    # Loss functions: Weighted CE on sarcasm strictly computed on train
    sarc_weights = compute_sarcasm_class_weights(train_df["sarcasm"]).to(config.device)
    ce_sarc = nn.CrossEntropyLoss(weight=sarc_weights)
    ce_sent = nn.CrossEntropyLoss()
    ce_dial = nn.CrossEntropyLoss()

    total_steps = len(train_loader) * config.max_epochs
    optimizer, scheduler = get_optimizer_and_scheduler(model, config, total_steps)

    use_fp16 = config.fp16 and ("cuda" in config.device)
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)
    early_stopper = EarlyStopping(patience=config.patience)

    start_time = time.perf_counter()

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{config.max_epochs}", disable=not verbose, leave=False)

        for step, batch in enumerate(pbar):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(config.device)
            attention_mask = batch["attention_mask"].to(config.device)
            y_sarc = batch["y_sarc"].to(config.device)

            with torch.amp.autocast(device_type="cuda" if "cuda" in config.device else "cpu", enabled=use_fp16):
                if config.task_mode == "mono":
                    logits_sarc = model(input_ids, attention_mask)
                    loss = ce_sarc(logits_sarc, y_sarc)
                else:
                    y_sent = batch["y_sent"].to(config.device)
                    if config.use_dialect:
                        logits_sarc, logits_sent, logits_dial = model(input_ids, attention_mask)
                        y_dial = batch["y_dial"].to(config.device)
                        loss = (
                            ce_sarc(logits_sarc, y_sarc)
                            + config.lam_sent * ce_sent(logits_sent, y_sent)
                            + config.lam_dial * ce_dial(logits_dial, y_dial)
                        )
                    else:
                        logits_sarc, logits_sent = model(input_ids, attention_mask)
                        loss = ce_sarc(logits_sarc, y_sarc) + config.lam_sent * ce_sent(logits_sent, y_sent)

            scaler.scale(loss).backward()
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            scale_after = scaler.get_scale()
            if scale_after >= scale_before:
                scheduler.step()

            running_loss += loss.item()
            pbar.set_postfix({"loss": f"{running_loss / (step + 1):.4f}"})

        # Validation at epoch end
        val_metrics = evaluate_model(model, val_loader, config)
        avg_loss = running_loss / len(train_loader)

        if verbose:
            print(
                f"Epoch {epoch}/{config.max_epochs} | Train Loss: {avg_loss:.4f} | "
                f"Val F1-Sarc: {val_metrics['val_f1_sarcasm']:.4f} (at thr={val_metrics['threshold']:.2f}) | "
                f"Val PR-AUC: {val_metrics['val_pr_auc']:.4f} | Val Macro-F1: {val_metrics['val_macro_f1']:.4f}"
            )

        # Early stopping check
        is_best = early_stopper.step(val_metrics["val_f1_sarcasm"], val_metrics["threshold"], model)
        if is_best and verbose:
            print(f"  -> New best validation F1 checkpoint saved: {val_metrics['val_f1_sarcasm']:.4f}")

        if early_stopper.early_stop:
            if verbose:
                print(f"Early stopping triggered at epoch {epoch} (patience={config.patience} reached).")
            break

    # Restore best checkpoint
    early_stopper.restore_best(model)
    best_thresh = early_stopper.best_threshold
    final_metrics = evaluate_model(model, val_loader, config, threshold=best_thresh)
    final_metrics["train_time_s"] = time.perf_counter() - start_time
    final_metrics["best_threshold"] = best_thresh
    final_metrics["n_trainable_params"] = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if verbose:
        print(f"\nFinal Restored Best Validation Metrics:")
        print(f"  Threshold: {best_thresh:.2f}")
        print(f"  F1-Sarcasm: {final_metrics['val_f1_sarcasm']:.4f}")
        print(f"  Macro-F1:   {final_metrics['val_macro_f1']:.4f}")
        print(f"  PR-AUC:     {final_metrics['val_pr_auc']:.4f}")
        print(f"  Accuracy:   {final_metrics['val_accuracy']:.4f}")

    return model, final_metrics


def compare_lambdas(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    base_config: TrainConfig,
    lambdas: list[float] = [0.3, 0.5, 1.0],
) -> pd.DataFrame:
    """Run E2 multi-task models with different lambda values and compare validation metrics."""
    records = []
    for lam in lambdas:
        print(f"\n{'='*25} Training with lambda = {lam} {'='*25}")
        cfg = copy.deepcopy(base_config)
        cfg.task_mode = "multi"
        cfg.lam_sent = lam
        _model, metrics = train_plm(train_df, val_df, cfg, verbose=True)
        records.append({
            "lambda": lam,
            "val_f1_sarcasm": metrics["val_f1_sarcasm"],
            "val_macro_f1": metrics["val_macro_f1"],
            "val_pr_auc": metrics["val_pr_auc"],
            "val_accuracy": metrics["val_accuracy"],
            "threshold": metrics["best_threshold"],
            "train_time_s": metrics["train_time_s"],
        })
    df_results = pd.DataFrame(records)
    print("\n" + "=" * 30 + " Lambda Comparison Summary " + "=" * 30)
    print(df_results.to_string(index=False))
    return df_results


def run_seeds(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    config: TrainConfig,
    seeds: list[int] = [42, 1337, 2024],
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Train 3 seeds per final model configuration and report mean +/- std."""
    records = []
    for seed in seeds:
        print(f"\n{'='*25} Running Seed {seed} {'='*25}")
        cfg = copy.deepcopy(config)
        cfg.seed = seed
        _model, metrics = train_plm(train_df, val_df, cfg, verbose=True)
        records.append({
            "seed": seed,
            "val_f1_sarcasm": metrics["val_f1_sarcasm"],
            "val_macro_f1": metrics["val_macro_f1"],
            "val_pr_auc": metrics["val_pr_auc"],
            "val_accuracy": metrics["val_accuracy"],
            "threshold": metrics["best_threshold"],
            "train_time_s": metrics["train_time_s"],
        })

    df_runs = pd.DataFrame(records)
    summary = {}
    for col in ["val_f1_sarcasm", "val_macro_f1", "val_pr_auc", "val_accuracy"]:
        mean = df_runs[col].mean()
        std = df_runs[col].std()
        summary[col] = f"{mean:.4f} ± {std:.4f}"

    print("\n" + "=" * 30 + " Multi-Seed Results (Mean ± Std) " + "=" * 30)
    for k, v in summary.items():
        print(f"{k}: {v}")

    return df_runs, summary

