"""Part B, headless: same functions as notebooks/part_b.ipynb.

No weights are updated. The official test file is not scored. A --limit run
prints metrics on a validation prefix and does not touch results.csv.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data import load_and_split  # noqa: E402
from evaluate import append_results, build_part_b_rows  # noqa: E402
from models.plm import (  # noqa: E402
    collect_part_b,
    hypothesis_frame,
    load_nli_pipeline,
    load_sentiment_pipeline,
    slice_validation,
)
from train import DEFAULT_SEED, set_seed  # noqa: E402


def _print_frame(title: str, frame) -> None:
    print(f"\n== {title} ==")
    print(frame.to_string(index=False))


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

    parser = argparse.ArgumentParser(description="Partie B — PLM sans entraînement")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="préfixe de la validation pour un essai court ; n'écrit pas results.csv",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    set_seed(DEFAULT_SEED)
    bundle = load_and_split()
    val = slice_validation(bundle["val"], args.limit)
    y_val = val["sarcasm"].astype(bool)
    print(f"validation scorée : {len(val)} tweets")
    if args.limit is not None:
        print("essai court : results.csv ne sera pas modifié")

    _print_frame("hypothèses NLI", hypothesis_frame())

    nli = load_nli_pipeline()
    sentiment = load_sentiment_pipeline()
    print("\n== id2label sentiment ==")
    print(sentiment.model.config.id2label)

    report = collect_part_b(val, nli, sentiment, batch_size=args.batch_size)
    rows = build_part_b_rows(
        y_val,
        [(run.name, run.pred, run.proba, run.threshold, run.elapsed_s) for run in report.nli],
        report.sentiment_pred,
        report.sentiment_proba,
        report.sentiment_elapsed_s,
        seed=DEFAULT_SEED,
    )
    _print_frame("partie B", pd.DataFrame(rows))

    if args.limit is None:
        saved = append_results(rows, ROOT / "results" / "results.csv")
        _print_frame("results/results.csv", saved)
        print("Prédictions de test non écrites : elles sont réservées à la partie F.")
    else:
        print("Prédictions de test non écrites. Journal inchangé à cause de --limit.")


if __name__ == "__main__":
    main()
