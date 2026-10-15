"""Shared Part C training budget, early stopping, and the imbalance comparison.

Early stopping watches sarcastic F1 on the validation split at threshold 0.5.
The official test file is never scored here.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from evaluate import choose_threshold, predict_at_threshold, sarcasm_f1
from models.deep import (
    D_MODEL,
    N_HEADS,
    N_LAYERS,
    build_classifier,
    count_trainable,
    embedding_from_matrix,
)
from models.static import (
    ARAVEC_DIM,
    MIN_FREQ,
    Vocab,
    aravec_token_coverage,
    build_char_vocab,
    build_embedding_matrix,
    build_vocab,
    char_ids_for_text,
    ids_for_text,
    load_aravec,
    oov_table,
    random_embedding_matrix,
    tokenize,
)

ARCHITECTURES = ("c1", "c2", "c3", "c3_no_pe")
ARCH_LABELS = {
    "c1": "c1_bilstm",
    "c2": "c2_attn",
    "c3": "c3_transformer",
    "c3_no_pe": "c3_transformer_no_pe",
    "c4": "c4_charcnn",
}
DECISION_THRESHOLD = 0.5

ARCH_NOTE = "early stopping sur val F1 sarcastique au seuil 0.5 ; test non utilisé"
WEIGHTED_NOTE = "perte pondérée calculée sur train ; seuil 0.5 ; test non utilisé"
OVERSAMPLE_NOTE = (
    "sur-échantillonnage de la classe sarcastique dans train uniquement ; seuil 0.5 ; test non utilisé"
)
THRESHOLD_NOTE = (
    "perte standard du run retenu ; seuil choisi sur val par F1 sarcastique ; test non utilisé"
)
FOCAL_NOTE = "focal loss ; seuil 0.5 ; test non utilisé"
CHAR_NOTE = (
    "char-CNN optionnel ; même budget d'entraînement ; "
    "early stopping sur val F1 sarcastique au seuil 0.5 ; test non utilisé"
)


@dataclass
class TrainConfig:
    """One budget for every architecture. Model selection never reads the official test."""

    lr: float = 1e-3
    batch_size: int = 64
    max_epochs: int = 8
    patience: int = 2
    dropout: float = 0.3
    hidden: int = 128
    embed_dim: int = 100
    d_model: int = D_MODEL
    nhead: int = N_HEADS
    num_layers: int = N_LAYERS
    max_len: int = 64
    max_char_len: int = 200
    pool: str = "max"
    grad_clip: float = 1.0
    decision_threshold: float = DECISION_THRESHOLD
    seed: int = 42


@dataclass
class PreparedPartC:
    train: pd.DataFrame
    val: pd.DataFrame
    vocab: Vocab
    oov: pd.DataFrame


@dataclass
class FitResult:
    proba: np.ndarray
    pred: np.ndarray
    threshold: float
    train_time_s: float
    n_trainable_params: int
    best_epoch: int
    best_val_f1: float
    model: nn.Module


@dataclass
class PartCReport:
    vocab_size: int
    oov: pd.DataFrame
    aravec_loaded: bool
    aravec_path: str
    coverage: dict[str, float | int] | None
    runs: list[dict[str, object]]
    attention: list[dict[str, object]] = field(default_factory=list)
    best_model: str = ""


class FocalLoss(nn.Module):
    """Optional focal loss. gamma = 2, no extra class weight."""

    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(reduction="none")

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = self.ce(logits, target)
        pt = torch.exp(-loss)
        return (((1.0 - pt) ** self.gamma) * loss).mean()


class _TextDataset(Dataset):
    def __init__(self, texts, labels, vocab: Vocab, max_len: int, char_level: bool):
        self.labels = _as_int_labels(labels)
        encode = char_ids_for_text if char_level else ids_for_text
        self.rows = [encode(text, vocab, max_len) for text in texts]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        return self.rows[index], int(self.labels[index])


def _as_int_labels(labels) -> np.ndarray:
    return np.asarray(labels).astype(bool).astype(np.int64)


def _as_bool_labels(labels) -> np.ndarray:
    return np.asarray(labels).astype(bool)


def slice_frame(frame: pd.DataFrame, limit: int | None) -> pd.DataFrame:
    if limit is None:
        return frame.reset_index(drop=True)
    return frame.iloc[: int(limit)].reset_index(drop=True)


def class_weights(labels) -> torch.Tensor:
    """Balanced weights n / (2 * count), computed on the labels that are passed in."""
    counts = np.bincount(_as_int_labels(labels), minlength=2).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    weights = len(np.asarray(labels)) / (2.0 * counts)
    return torch.tensor(weights, dtype=torch.float32)


def oversample_minority(texts, labels, seed: int) -> tuple[list[str], np.ndarray]:
    """Duplicate sarcastic rows until they match the other class.

    Validation is not an argument: this function only sees the split it is given,
    which the caller must restrict to train.
    """
    stored = [str(text) for text in texts]
    encoded = _as_int_labels(labels)
    positive = np.flatnonzero(encoded == 1)
    negative = np.flatnonzero(encoded == 0)
    if len(positive) == 0 or len(negative) == 0 or len(positive) >= len(negative):
        return stored, encoded
    rng = np.random.default_rng(seed)
    extra = rng.choice(positive, size=len(negative) - len(positive), replace=True)
    index = np.concatenate([np.arange(len(encoded)), extra])
    rng.shuffle(index)
    return [stored[i] for i in index], encoded[index]


def early_stop_update(val_f1: float, best_f1: float, wait: int, patience: int) -> tuple[float, int, bool, bool]:
    """Returns the new best, the wait count, whether this epoch won, and whether to stop."""
    if val_f1 > best_f1:
        return val_f1, 0, True, False
    wait += 1
    return best_f1, wait, False, wait >= patience


def prepare_part_c(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, limit: int | None = None) -> PreparedPartC:
    """Train vocabulary and descriptive OOV rates. The test split is not kept for training."""
    train = slice_frame(train_df, limit)
    val = slice_frame(val_df, limit)
    test = slice_frame(test_df, limit)
    vocab = build_vocab(train["tweet"], min_freq=MIN_FREQ)
    oov = oov_table(
        {"train": train["tweet"], "val": val["tweet"], "test": test["tweet"]},
        vocab,
    )
    return PreparedPartC(train=train, val=val, vocab=vocab, oov=oov)


def _collate(batch):
    rows, labels = zip(*batch)
    width = max(len(row) for row in rows)
    ids = torch.zeros(len(rows), width, dtype=torch.long)
    mask = torch.zeros(len(rows), width, dtype=torch.bool)
    for index, row in enumerate(rows):
        ids[index, : len(row)] = torch.tensor(row, dtype=torch.long)
        mask[index, : len(row)] = True
    return ids, mask, torch.tensor(labels, dtype=torch.long)


def _loader(texts, labels, vocab: Vocab, config: TrainConfig, shuffle: bool, char_level: bool) -> DataLoader:
    max_len = config.max_char_len if char_level else config.max_len
    dataset = _TextDataset(texts, labels, vocab, max_len, char_level)
    generator = None
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(config.seed)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        generator=generator,
        collate_fn=_collate,
    )


def _make_loss(loss_name: str, labels, device: torch.device) -> nn.Module:
    if loss_name == "standard":
        return nn.CrossEntropyLoss()
    if loss_name == "weighted":
        return nn.CrossEntropyLoss(weight=class_weights(labels).to(device))
    if loss_name == "focal":
        return FocalLoss()
    raise ValueError(f"unknown loss {loss_name!r}")


def train_epoch(model, loader, criterion, optimizer, device: torch.device, grad_clip: float) -> None:
    model.train()
    for ids, mask, target in loader:
        ids = ids.to(device)
        mask = mask.to(device)
        target = target.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(ids, mask), target)
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()


@torch.no_grad()
def predict_proba(model, loader, device: torch.device) -> np.ndarray:
    model.eval()
    chunks = []
    for ids, mask, _target in loader:
        logits = model(ids.to(device), mask.to(device))
        chunks.append(torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy())
    if not chunks:
        return np.zeros(0, dtype=np.float64)
    return np.concatenate(chunks).astype(np.float64)


def fit_model(
    model: nn.Module,
    train_texts,
    train_labels,
    val_texts,
    val_labels,
    vocab: Vocab,
    config: TrainConfig,
    *,
    loss_name: str = "standard",
    oversample: bool = False,
    char_level: bool = False,
) -> FitResult:
    """Train with the shared budget and restore the best validation checkpoint."""
    from train import set_seed

    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    labels = _as_int_labels(train_labels)
    texts = list(train_texts)
    loss_labels = labels
    if oversample:
        texts, labels = oversample_minority(texts, labels, config.seed)
    criterion = _make_loss(loss_name, loss_labels, device)
    optimizer = torch.optim.Adam((parameter for parameter in model.parameters() if parameter.requires_grad), lr=config.lr)
    train_loader = _loader(texts, labels, vocab, config, shuffle=True, char_level=char_level)
    val_loader = _loader(val_texts, val_labels, vocab, config, shuffle=False, char_level=char_level)
    val_bool = _as_bool_labels(val_labels)

    best_f1 = -1.0
    wait = 0
    best_state = None
    best_proba = None
    best_pred = None
    best_epoch = 0
    started = time.perf_counter()
    for epoch in range(1, config.max_epochs + 1):
        train_epoch(model, train_loader, criterion, optimizer, device, config.grad_clip)
        proba = predict_proba(model, val_loader, device)
        pred = proba >= config.decision_threshold
        val_f1 = sarcasm_f1(val_bool, pred)
        best_f1, wait, improved, stop = early_stop_update(val_f1, best_f1, wait, config.patience)
        if improved:
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_proba = proba
            best_pred = pred
            best_epoch = epoch
        if stop:
            break
    if best_state is None or best_proba is None or best_pred is None:
        raise RuntimeError("aucune époque n'a été retenue ; max_epochs doit valoir au moins 1")
    model.load_state_dict(best_state)
    elapsed = time.perf_counter() - started
    return FitResult(
        proba=np.asarray(best_proba),
        pred=np.asarray(best_pred),
        threshold=config.decision_threshold,
        train_time_s=elapsed,
        n_trainable_params=count_trainable(model),
        best_epoch=best_epoch,
        best_val_f1=float(best_f1),
        model=model,
    )


def _run_name(arch: str, kind: str, freeze: bool, dim: int, suffix: str = "") -> str:
    if arch == "c4":
        base = ARCH_LABELS["c4"]
    elif kind == "random":
        base = f"{ARCH_LABELS[arch]}_random"
    else:
        state = "frozen" if freeze else "tuned"
        base = f"{ARCH_LABELS[arch]}_aravec{dim}_{state}"
    return f"{base}_{suffix}" if suffix else base


def _preprocess_name(kind: str) -> str:
    if kind == "char":
        return "char"
    if kind == "aravec":
        return "word_min2+aravec_clean_str"
    return "word_min2"


def _arch_note(arch: str, best_epoch: int, max_epochs: int, pool: str) -> str:
    detail = {
        "c1": f"BiLSTM {pool}-pooling",
        "c2": "BiLSTM + attention additive",
        "c3": "mini-Transformer avec encodage positionnel",
        "c3_no_pe": "ablation sans encodage positionnel",
    }[arch]
    return f"{detail} ; {ARCH_NOTE} ; meilleure époque {best_epoch}/{max_epochs}"


def _make_word_model(arch: str, vocab: Vocab, config: TrainConfig, kind: str, freeze: bool, matrix: np.ndarray | None):
    embedding = None
    embed_dim = config.embed_dim
    if kind == "aravec":
        if matrix is None:
            raise ValueError("AraVec matrix is missing")
        embedding = embedding_from_matrix(matrix, freeze=freeze)
        embed_dim = int(matrix.shape[1])
    else:
        embedding = embedding_from_matrix(
            random_embedding_matrix(len(vocab), config.embed_dim, config.seed),
            freeze=False,
        )
    return build_classifier(
        arch,
        len(vocab),
        embed_dim,
        hidden=config.hidden,
        dropout=config.dropout,
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
        pool=config.pool,
        max_len=config.max_len,
        embedding=embedding,
    )


def _run_dict(
    name: str,
    preprocess: str,
    result: FitResult,
    notes: str,
    role: str,
    order: int,
    arch: str,
    kind: str,
    freeze: bool,
) -> dict[str, object]:
    return {
        "model": name,
        "preprocess": preprocess,
        "pred": result.pred,
        "proba": result.proba,
        "threshold": result.threshold,
        "train_time_s": result.train_time_s,
        "n_trainable_params": result.n_trainable_params,
        "notes": notes,
        "role": role,
        "val_f1": result.best_val_f1,
        "order": order,
        "arch": arch,
        "kind": kind,
        "freeze": freeze,
    }


def attention_examples(model, texts, labels, vocab: Vocab, config: TrainConfig, n: int = 5) -> list[dict[str, object]]:
    """Attention weights of a C2 model on the first sarcastic validation tweets."""
    chosen = [text for text, label in zip(texts, labels) if bool(label)][:n]
    if not chosen:
        return []
    device = next(model.parameters()).device
    model.eval()
    rows = []
    for text in chosen:
        ids = ids_for_text(text, vocab, config.max_len)
        tokens = tokenize(text)[: config.max_len] or ["<unk>"]
        tensor = torch.tensor([ids], dtype=torch.long, device=device)
        mask = torch.ones(1, len(ids), dtype=torch.bool, device=device)
        with torch.no_grad():
            _logits, alpha = model(tensor, mask, return_attention=True)
        weights = [round(float(value), 4) for value in alpha.squeeze(0).detach().cpu().tolist()]
        rows.append({"tweet": text, "tokens": tokens, "weights": weights})
    return rows


def attention_frame(examples: list[dict[str, object]]) -> pd.DataFrame:
    rows = []
    for example in examples:
        pairs = " ".join(
            f"{token}:{weight:.2f}" for token, weight in zip(example["tokens"], example["weights"])
        )
        rows.append({"tweet": example["tweet"], "attention": pairs})
    return pd.DataFrame(rows)


def collect_part_c(
    prepared: PreparedPartC,
    config: TrainConfig | None = None,
    *,
    skip_aravec: bool = False,
    char_cnn: bool = False,
    focal: bool = False,
) -> PartCReport:
    """Random and AraVec runs of C1–C3, then the imbalance comparison on the best one.

    C3 includes the positional-encoding ablation. C4 and focal loss stay optional.
    """
    from train import set_seed

    config = config or TrainConfig()
    set_seed(config.seed)
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"appareil : {device_name}")
    print(
        f"budget : lr {config.lr:g}, batch {config.batch_size}, "
        f"{config.max_epochs} époques, patience {config.patience}, hidden {config.hidden}"
    )

    matrix = None
    coverage = None
    aravec_path = ""
    if skip_aravec:
        print("AraVec écarté (--skip-aravec).")
    else:
        vectors = load_aravec(ARAVEC_DIM)
        if vectors is None:
            print(
                "AraVec introuvable : déposez full_uni_cbow_"
                f"{ARAVEC_DIM}_twitter.mdl dans data/aravec/ "
                "(Twitter, unigrammes, CBOW). Les runs aléatoires continuent."
            )
        else:
            aravec_path = str(vectors.path)
            matrix, coverage = build_embedding_matrix(prepared.vocab, vectors, vectors.dim, config.seed)
            token_coverage = aravec_token_coverage(prepared.val["tweet"], vectors)
            coverage = {**coverage, "val_token_hit_rate": token_coverage["hit_rate"]}
            print(f"AraVec : {vectors.path} ({vectors.dim} dim.)")
            print(
                f"couverture vocabulaire {coverage['vocab_hit_rate']:.3f} ; "
                f"tokens de val {coverage['val_token_hit_rate']:.3f}"
            )

    variants: list[tuple[str, np.ndarray | None, bool]] = [("random", None, False)]
    if matrix is not None:
        variants.append(("aravec", matrix, True))
        variants.append(("aravec", matrix, False))

    runs: list[dict[str, object]] = []
    attention: list[dict[str, object]] = []
    best_c2_f1 = -1.0
    order = 0
    train_tweets = prepared.train["tweet"]
    train_labels = prepared.train["sarcasm"]
    val_tweets = prepared.val["tweet"]
    val_labels = prepared.val["sarcasm"]
    dim = int(matrix.shape[1]) if matrix is not None else config.embed_dim

    for arch in ARCHITECTURES:
        for kind, _init, freeze in variants:
            order += 1
            set_seed(config.seed)
            model = _make_word_model(arch, prepared.vocab, config, kind, freeze, matrix)
            result = fit_model(
                model,
                train_tweets,
                train_labels,
                val_tweets,
                val_labels,
                prepared.vocab,
                config,
            )
            name = _run_name(arch, kind, freeze, dim)
            run = _run_dict(
                name,
                _preprocess_name(kind),
                result,
                _arch_note(arch, result.best_epoch, config.max_epochs, config.pool),
                "architecture",
                order,
                arch,
                kind,
                freeze,
            )
            runs.append(run)
            print(
                f"{name} — F1 val {result.best_val_f1:.4f} — "
                f"époque {result.best_epoch} — {result.train_time_s:.1f} s — "
                f"{result.n_trainable_params} paramètres"
            )
            if arch == "c2" and result.best_val_f1 > best_c2_f1:
                best_c2_f1 = result.best_val_f1
                attention = attention_examples(result.model, val_tweets, val_labels, prepared.vocab, config)

    architecture_runs = [run for run in runs if run["role"] == "architecture"]
    best = max(architecture_runs, key=lambda run: (float(run["val_f1"]), -int(run["order"])))
    print(f"meilleur C1–C3 : {best['model']}")

    for loss_name, oversample, suffix, note, role in (
        ("weighted", False, "weighted", WEIGHTED_NOTE, "weighted"),
        ("standard", True, "oversample", OVERSAMPLE_NOTE, "oversample"),
    ):
        order += 1
        set_seed(config.seed)
        model = _make_word_model(
            str(best["arch"]),
            prepared.vocab,
            config,
            str(best["kind"]),
            bool(best["freeze"]),
            matrix,
        )
        result = fit_model(
            model,
            train_tweets,
            train_labels,
            val_tweets,
            val_labels,
            prepared.vocab,
            config,
            loss_name=loss_name,
            oversample=oversample,
        )
        name = f"{best['model']}_{suffix}"
        run = _run_dict(
            name,
            str(best["preprocess"]),
            result,
            f"{note} ; meilleure époque {result.best_epoch}/{config.max_epochs}",
            role,
            order,
            str(best["arch"]),
            str(best["kind"]),
            bool(best["freeze"]),
        )
        runs.append(run)
        print(f"{name} — F1 val {result.best_val_f1:.4f} — {result.train_time_s:.1f} s")

    y_val = _as_bool_labels(val_labels)
    threshold, threshold_f1 = choose_threshold(y_val, best["proba"])
    order += 1
    runs.append(
        {
            "model": f"{best['model']}_threshold",
            "preprocess": best["preprocess"],
            "pred": predict_at_threshold(best["proba"], threshold),
            "proba": best["proba"],
            "threshold": threshold,
            "train_time_s": 0.0,
            "n_trainable_params": best["n_trainable_params"],
            "notes": THRESHOLD_NOTE,
            "role": "threshold",
            "val_f1": threshold_f1,
            "order": order,
            "arch": best["arch"],
            "kind": best["kind"],
            "freeze": best["freeze"],
        }
    )
    print(f"{best['model']}_threshold — seuil {threshold:.2f} — F1 val {threshold_f1:.4f}")

    if focal:
        order += 1
        set_seed(config.seed)
        model = _make_word_model(
            str(best["arch"]),
            prepared.vocab,
            config,
            str(best["kind"]),
            bool(best["freeze"]),
            matrix,
        )
        result = fit_model(
            model,
            train_tweets,
            train_labels,
            val_tweets,
            val_labels,
            prepared.vocab,
            config,
            loss_name="focal",
        )
        name = f"{best['model']}_focal"
        runs.append(
            _run_dict(
                name,
                str(best["preprocess"]),
                result,
                f"{FOCAL_NOTE} ; meilleure époque {result.best_epoch}/{config.max_epochs}",
                "focal",
                order,
                str(best["arch"]),
                str(best["kind"]),
                bool(best["freeze"]),
            )
        )
        print(f"{name} — F1 val {result.best_val_f1:.4f} — {result.train_time_s:.1f} s")

    if char_cnn:
        order += 1
        char_vocab = build_char_vocab(train_tweets)
        set_seed(config.seed)
        model = build_classifier(
            "c4",
            len(char_vocab),
            config.embed_dim,
            hidden=config.hidden,
            dropout=config.dropout,
            d_model=config.d_model,
            nhead=config.nhead,
            num_layers=config.num_layers,
            pool=config.pool,
            max_len=config.max_char_len,
        )
        result = fit_model(
            model,
            train_tweets,
            train_labels,
            val_tweets,
            val_labels,
            char_vocab,
            config,
            char_level=True,
        )
        name = _run_name("c4", "char", False, dim)
        runs.append(
            _run_dict(
                name,
                _preprocess_name("char"),
                result,
                f"{CHAR_NOTE} ; meilleure époque {result.best_epoch}/{config.max_epochs}",
                "optional",
                order,
                "c4",
                "char",
                False,
            )
        )
        print(f"{name} — F1 val {result.best_val_f1:.4f} — {result.train_time_s:.1f} s")

    return PartCReport(
        vocab_size=len(prepared.vocab),
        oov=prepared.oov,
        aravec_loaded=matrix is not None,
        aravec_path=aravec_path,
        coverage=coverage,
        runs=runs,
        attention=attention,
        best_model=str(best["model"]),
    )
