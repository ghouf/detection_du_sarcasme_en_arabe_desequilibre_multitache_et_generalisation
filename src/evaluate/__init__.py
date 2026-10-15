"""Validation metrics, the decision threshold, the results journal, and prediction files.

Test metrics stay empty until Part F. ``export_predictions`` is ready for Part F
and is not called on the official test here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, f1_score

from preprocess import add_surface_features

RESULTS_COLUMNS = [
    "model",
    "preprocess",
    "seed",
    "val_f1_sarcasm",
    "val_pr_auc",
    "val_macro_f1",
    "val_accuracy",
    "test_f1_sarcasm",
    "test_pr_auc",
    "test_macro_f1",
    "threshold",
    "train_time_s",
    "n_trainable_params",
    "notes",
]

RULE_FEATURES = (
    ("rule_url", "has_url"),
    ("rule_mention", "has_mention"),
    ("rule_csv_quote", "has_csv_quote"),
)


def sarcasm_f1(y_true, y_pred) -> float:
    return float(f1_score(y_true, y_pred, pos_label=True, zero_division=0))


def macro_f1(y_true, y_pred) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def pr_auc(y_true, proba) -> float:
    return float(average_precision_score(np.asarray(y_true).astype(int), np.asarray(proba)))


def accuracy(y_true, y_pred) -> float:
    return float(accuracy_score(y_true, y_pred))


def choose_threshold(y_true, proba, grid: np.ndarray | None = None) -> tuple[float, float]:
    """Lowest threshold on the grid that maximizes sarcastic F1."""
    if grid is None:
        grid = np.round(np.arange(0.01, 1.00, 0.01), 2)
    labels = np.asarray(y_true).astype(bool)
    scores = np.asarray(proba, dtype=float)
    best_threshold = float(grid[0])
    best_f1 = -1.0
    for threshold in grid:
        f1 = sarcasm_f1(labels, scores >= threshold)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)
    return best_threshold, best_f1


def trivial_predict(n_rows: int) -> tuple[np.ndarray, np.ndarray]:
    """Always predict non-sarcastic. Probability of sarcasm is 0."""
    pred = np.zeros(n_rows, dtype=bool)
    proba = np.zeros(n_rows, dtype=float)
    return pred, proba


def predict_at_threshold(proba, threshold: float) -> np.ndarray:
    return np.asarray(proba) >= threshold


def sarcasm_rate_table(splits: dict[str, pd.DataFrame], by: str | None = None) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for name, frame in splits.items():
        labels = frame["sarcasm"].astype(bool)
        if by is None:
            frames.append(
                pd.DataFrame(
                    {
                        "split": [name],
                        "n": [len(frame)],
                        "n_sarcastic": [int(labels.sum())],
                        "sarcasm_rate": [float(labels.mean()) if len(frame) else float("nan")],
                    }
                )
            )
            continue
        grouped = frame.assign(_label=labels).groupby(by, dropna=False)["_label"]
        part = grouped.agg(n="size", n_sarcastic="sum", sarcasm_rate="mean").reset_index()
        part.insert(0, "split", name)
        frames.append(part)
    return pd.concat(frames, ignore_index=True)


def fit_single_rule(train_df: pd.DataFrame, feature_col: str) -> dict[str, object]:
    """Choose the rule direction from the train sarcasm rate of this feature."""
    present = train_df[feature_col].astype(bool)
    labels = train_df["sarcasm"].astype(bool)
    rate_present = float(labels[present].mean()) if present.any() else float("nan")
    rate_absent = float(labels[~present].mean()) if (~present).any() else float("nan")
    if np.isnan(rate_present):
        positive_when_present = False
    elif np.isnan(rate_absent):
        positive_when_present = True
    else:
        positive_when_present = bool(rate_present >= rate_absent)
    return {
        "feature": feature_col,
        "positive_when_present": positive_when_present,
        "train_rate_present": rate_present,
        "train_rate_absent": rate_absent,
    }


def apply_single_rule(df: pd.DataFrame, rule: dict[str, object]) -> tuple[np.ndarray, np.ndarray]:
    present = df[rule["feature"]].astype(bool).to_numpy()
    pred = present if rule["positive_when_present"] else ~present
    return pred, pred.astype(float)


def metric_row(
    model: str,
    preprocess: str,
    seed: int,
    y_true,
    y_pred,
    proba,
    threshold: float | None,
    train_time_s: float,
    n_trainable_params: int,
    notes: str,
    include_accuracy: bool = False,
) -> dict[str, object]:
    """One results.csv row. Test columns stay empty until Part F."""
    return {
        "model": model,
        "preprocess": preprocess,
        "seed": seed,
        "val_f1_sarcasm": round(sarcasm_f1(y_true, y_pred), 6),
        "val_pr_auc": round(pr_auc(y_true, proba), 6),
        "val_macro_f1": round(macro_f1(y_true, y_pred), 6),
        "val_accuracy": round(accuracy(y_true, y_pred), 6) if include_accuracy else "",
        "test_f1_sarcasm": "",
        "test_pr_auc": "",
        "test_macro_f1": "",
        "threshold": "" if threshold is None else round(float(threshold), 2),
        "train_time_s": round(float(train_time_s), 3),
        "n_trainable_params": int(n_trainable_params),
        "notes": notes,
    }


def append_results(rows: list[dict[str, object]], path: Path) -> pd.DataFrame:
    """Write ``rows``, replacing earlier rows that share the same model name."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fresh = pd.DataFrame(rows, columns=RESULTS_COLUMNS)
    if path.exists() and path.stat().st_size > 0:
        previous = pd.read_csv(path)
        if "model" in previous.columns and not previous.empty:
            previous = previous[~previous["model"].isin(fresh["model"])]
            fresh = pd.concat([previous, fresh], ignore_index=True)
    fresh = fresh.reindex(columns=RESULTS_COLUMNS)
    fresh.to_csv(path, index=False)
    return fresh


def export_predictions(path: Path, ids, y_true, y_pred, proba) -> pd.DataFrame:
    """Part F file: id, y_true, y_pred, proba_sarcasm. Not used in Part A."""
    frame = pd.DataFrame(
        {
            "id": np.asarray(ids),
            "y_true": np.asarray(y_true).astype(bool),
            "y_pred": np.asarray(y_pred).astype(bool),
            "proba_sarcasm": np.asarray(proba, dtype=float),
        }
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame


def prediction_path(directory: Path, model: str, seed: int) -> Path:
    return Path(directory) / f"{model}_seed{seed}.csv"


def evaluate_rules(train_df: pd.DataFrame, val_df: pd.DataFrame) -> tuple[pd.DataFrame, list[tuple[str, np.ndarray, np.ndarray]]]:
    """Fit each one-feature rule on train and score it on val."""
    train_feat = add_surface_features(train_df)
    val_feat = add_surface_features(val_df)
    y_val = val_feat["sarcasm"].astype(bool)
    summary: list[dict[str, object]] = []
    predictions: list[tuple[str, np.ndarray, np.ndarray]] = []
    for name, column in RULE_FEATURES:
        rule = fit_single_rule(train_feat, column)
        pred, proba = apply_single_rule(val_feat, rule)
        summary.append(
            {
                "model": name,
                "feature": column,
                "positive_when_present": rule["positive_when_present"],
                "train_rate_present": rule["train_rate_present"],
                "train_rate_absent": rule["train_rate_absent"],
                "val_f1_sarcasm": sarcasm_f1(y_val, pred),
                "val_pr_auc": pr_auc(y_val, proba),
                "val_macro_f1": macro_f1(y_val, pred),
            }
        )
        predictions.append((name, pred, proba))
    return pd.DataFrame(summary), predictions


def build_part_a_rows(
    y_val,
    baseline_pred,
    baseline_proba,
    threshold: float,
    train_time_s: float,
    n_params: int,
    rule_predictions: list[tuple[str, np.ndarray, np.ndarray]],
    seed: int = 42,
) -> list[dict[str, object]]:
    """Validation rows for Part A. Test columns are left empty on purpose."""
    trivial_pred, trivial_proba = trivial_predict(len(np.asarray(y_val)))
    rows = [
        metric_row(
            "trivial_never_sarcastic",
            "none",
            seed,
            y_val,
            trivial_pred,
            trivial_proba,
            None,
            0.0,
            0,
            "toujours non sarcastique ; l'accuracy n'est pas la métrique de sélection",
            include_accuracy=True,
        ),
        metric_row(
            "tfidf_word12_char25_logreg",
            "strip_csv_quotes",
            seed,
            y_val,
            baseline_pred,
            baseline_proba,
            threshold,
            train_time_s,
            n_params,
            "seuil choisi sur val par F1 sarcastique maximal ; test non utilisé",
        ),
    ]
    for name, pred, proba in rule_predictions:
        rows.append(
            metric_row(
                name,
                "raw",
                seed,
                y_val,
                pred,
                proba,
                0.5,
                0.0,
                0,
                "direction choisie sur le taux de sarcasme du train",
            )
        )
    return rows


NLI_NOTE = (
    "aucun entraînement ; seuil de P(sarcastique) choisi sur val ; "
    "train_time_s est le temps d'inférence ; test non utilisé"
)
SENTIMENT_NOTE = (
    "règle sentiment négatif => sarcastique ; décision par argmax ; "
    "fine-tuné sur ASTD, ArSAS et SemEval ; ArSarcasm v1 ré-annote ASTD et SemEval 2017, "
    "donc certains tweets ont pu être vus ; train_time_s est le temps d'inférence ; test non utilisé"
)


def build_part_b_rows(
    y_val,
    nli_runs: list[tuple[str, np.ndarray, np.ndarray, float, float]],
    sentiment_pred,
    sentiment_proba,
    sentiment_time_s: float,
    seed: int = 42,
) -> list[dict[str, object]]:
    """Validation rows for Part B. Test columns are left empty on purpose.

    Each NLI item is ``(model, pred, proba, threshold, inference_seconds)``.
    The sentiment row has no threshold: the decision is the model's argmax.
    """
    rows = [
        metric_row(
            name,
            "raw",
            seed,
            y_val,
            pred,
            proba,
            threshold,
            elapsed,
            0,
            NLI_NOTE,
        )
        for name, pred, proba, threshold, elapsed in nli_runs
    ]
    rows.append(
        metric_row(
            "sentiment_neg_implies_sarcasm",
            "raw",
            seed,
            y_val,
            sentiment_pred,
            sentiment_proba,
            None,
            sentiment_time_s,
            0,
            SENTIMENT_NOTE,
        )
    )
    return rows


def build_part_c_rows(y_val, runs: list[dict[str, object]], seed: int = 42) -> list[dict[str, object]]:
    """Validation rows for Part C. Test columns stay empty on purpose.

    Each run provides model, preprocess, pred, proba, threshold, train_time_s,
    n_trainable_params, and notes. The official test file is not scored.
    """
    return [
        metric_row(
            str(run["model"]),
            str(run["preprocess"]),
            seed,
            y_val,
            run["pred"],
            run["proba"],
            None if run["threshold"] is None else float(run["threshold"]),
            float(run["train_time_s"]),
            int(run["n_trainable_params"]),
            str(run["notes"]),
        )
        for run in runs
    ]


_SLICE_COLORS = {
    "NEG": "#c44e52",
    "NEU": "#8c8c8c",
    "POS": "#4c72b0",
    "msa": "#4c72b0",
    "egypt": "#dd8452",
    "gulf": "#55a868",
    "levant": "#c44e52",
    "magreb": "#8172b3",
}


def circular_partition(splits: dict[str, pd.DataFrame], column: str, title: str, order: list[str] | None = None):
    """One pie per split for a categorical column (sentiment or dialect)."""
    import matplotlib
    import sys

    if "matplotlib.pyplot" not in sys.modules:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(splits)
    fig, axes = plt.subplots(1, len(names), figsize=(4.8 * max(len(names), 1), 5.6))
    axes = np.atleast_1d(axes)
    for ax, name in zip(axes, names):
        series = splits[name][column].astype(str)
        counts = series.value_counts()
        if order is not None:
            counts = counts.reindex(list(order)).fillna(0)
        counts = counts[counts > 0]
        colors = [_SLICE_COLORS.get(str(label), "#4c72b0") for label in counts.index]
        total = float(counts.sum())
        wedges, _texts = ax.pie(
            counts.to_numpy(),
            colors=colors,
            startangle=90,
            counterclock=False,
        )
        legend = [
            f"{label} — {100 * float(counts[label]) / total:.1f}%"
            for label in counts.index
        ]
        ax.legend(
            wedges,
            legend,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.02),
            frameon=False,
            fontsize=9,
        )
        ax.set_title(f"{name}\nn={len(series)}")
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return fig
