"""Part A, headless: same functions as notebooks/part_a.ipynb.

The official test file is used for sarcasm rates and for leakage removal.
It is not used to fit TF-IDF, to train the classifier, or to choose the threshold.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data import load_and_split, sample_raw_tweets  # noqa: E402
from evaluate import (  # noqa: E402
    accuracy,
    append_results,
    build_part_a_rows,
    choose_threshold,
    circular_partition,
    evaluate_rules,
    predict_at_threshold,
    sarcasm_f1,
    sarcasm_rate_table,
    trivial_predict,
)
from models.baseline import sarcastic_proba  # noqa: E402
from preprocess import audit_summary, duplicate_report  # noqa: E402
from train import DEFAULT_SEED, fit_baseline, set_seed  # noqa: E402


def _print_frame(title: str, frame) -> None:
    print(f"\n== {title} ==")
    print(frame.to_string(index=False))


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

    set_seed(DEFAULT_SEED)
    bundle = load_and_split()
    full = bundle["full"]
    test = bundle["test"]
    train = bundle["train"]
    val = bundle["val"]

    print(f"fichier train officiel : {len(full)}")
    print(f"fichier test officiel  : {len(test)}")
    print(f"train après nettoyage  : {len(train)}")
    print(f"val (split imposé)     : {len(val)}")
    print("fuites retirées du train :", bundle["leak"])
    print("labels contradictoires   :", bundle["contradictory"])

    train_sample = sample_raw_tweets(full, n=30, seed=DEFAULT_SEED)
    test_sample = sample_raw_tweets(test, n=30, seed=DEFAULT_SEED)
    print(f"échantillon brut train : {len(train_sample)} tweets (seed {DEFAULT_SEED})")
    print(f"échantillon brut test  : {len(test_sample)} tweets (seed {DEFAULT_SEED})")

    _print_frame("audit fichier train", audit_summary(full))
    _print_frame("audit fichier test", audit_summary(test))
    _print_frame(
        "doublons",
        pd.DataFrame(
            [
                duplicate_report(full, test, "training_file", "test"),
                duplicate_report(train, test, "train_split_cleaned", "test"),
            ]
        ),
    )

    splits = {"train": train, "val": val, "test": test}
    _print_frame("taux global", sarcasm_rate_table(splits))
    _print_frame("taux par dialecte", sarcasm_rate_table(splits, by="dialect"))
    _print_frame("taux par sentiment", sarcasm_rate_table(splits, by="sentiment"))

    figures = ROOT / "results"
    sentiment_fig = circular_partition(
        splits,
        "sentiment",
        "Répartition du sentiment (NEG / NEU / POS)",
        order=["NEG", "NEU", "POS"],
    )
    dialect_fig = circular_partition(
        splits,
        "dialect",
        "Répartition des dialectes",
        order=["msa", "egypt", "gulf", "levant", "magreb"],
    )
    sentiment_path = figures / "sentiment_partition.png"
    dialect_path = figures / "dialect_partition.png"
    sentiment_fig.savefig(sentiment_path, dpi=150, bbox_inches="tight")
    dialect_fig.savefig(dialect_path, dpi=150, bbox_inches="tight")
    plt.close(sentiment_fig)
    plt.close(dialect_fig)
    print(f"camemberts : {sentiment_path.name}, {dialect_path.name}")

    y_val = val["sarcasm"].astype(bool)
    trivial_pred, _trivial_proba = trivial_predict(len(val))
    trivial_f1 = sarcasm_f1(y_val, trivial_pred)
    trivial_acc = accuracy(y_val, trivial_pred)
    print(f"\nclassifieur trivial sur val — accuracy {trivial_acc:.4f} — F1 sarcastique {trivial_f1:.4f}")
    print(
        "L'accuracy est élevée parce que la classe sarcastique est minoritaire. "
        "Elle n'est pas une métrique acceptable. On retient le F1 de la classe sarcastique."
    )

    model, elapsed, n_params = fit_baseline(train["tweet"], train["sarcasm"], seed=DEFAULT_SEED)
    val_proba = sarcastic_proba(model, val["tweet"])
    threshold, best_f1 = choose_threshold(y_val, val_proba)
    val_pred = predict_at_threshold(val_proba, threshold)
    print(
        f"baseline TF-IDF — seuil {threshold:.2f} — F1 val {best_f1:.4f} — "
        f"{elapsed:.1f}s — {n_params} paramètres"
    )

    rule_table, rule_predictions = evaluate_rules(train, val)
    _print_frame("règles à un indice", rule_table)
    print(
        "Ces règles mesurent des raccourcis de surface. "
        "Un F1 proche de la baseline voudrait dire que le score exploite l'indice, "
        "pas le sarcasme. Le test officiel n'entre pas dans ce choix."
    )

    rows = build_part_a_rows(
        y_val,
        val_pred,
        val_proba,
        threshold,
        elapsed,
        n_params,
        rule_predictions,
        seed=DEFAULT_SEED,
    )
    saved = append_results(rows, ROOT / "results" / "results.csv")
    _print_frame("results/results.csv", saved)
    print("Prédictions de test non écrites : elles sont réservées à la partie F.")


if __name__ == "__main__":
    main()
