"""Part C, headless: same functions as notebooks/part_c.ipynb.

Early stopping and the decision threshold use validation sarcastic F1.
The official test file is not scored. A --limit run trains on a prefix and
does not touch results.csv.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data import load_and_split  # noqa: E402
from evaluate import append_results, build_part_c_rows  # noqa: E402
from train import DEFAULT_SEED, set_seed  # noqa: E402
from train.deep import (  # noqa: E402
    attention_frame,
    collect_part_c,
    prepare_part_c,
)


def _print_frame(title: str, frame) -> None:
    print(f"\n== {title} ==")
    if frame is None or len(frame) == 0:
        print("(vide)")
        return
    print(frame.to_string(index=False))


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

    parser = argparse.ArgumentParser(description="Partie C — embeddings statiques et réseaux")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="préfixe de train et de val pour un essai court ; n'écrit pas results.csv",
    )
    parser.add_argument(
        "--skip-aravec",
        action="store_true",
        help="ne pas charger AraVec, même si le fichier est présent",
    )
    parser.add_argument("--char-cnn", action="store_true", help="ajoute le char-CNN optionnel (C4)")
    parser.add_argument("--focal", action="store_true", help="ajoute la focal loss optionnelle")
    args = parser.parse_args()

    set_seed(DEFAULT_SEED)
    bundle = load_and_split()
    prepared = prepare_part_c(bundle["train"], bundle["val"], bundle["test"], limit=args.limit)
    print(f"vocabulaire train (fréquence ≥ 2) : {len(prepared.vocab)} types")
    if args.limit is not None:
        print("essai court : results.csv ne sera pas modifié")
    _print_frame("taux d'OOV", prepared.oov)

    report = collect_part_c(
        prepared,
        skip_aravec=args.skip_aravec,
        char_cnn=args.char_cnn,
        focal=args.focal,
    )
    y_val = prepared.val["sarcasm"].astype(bool)
    rows = build_part_c_rows(y_val, report.runs, seed=DEFAULT_SEED)
    _print_frame("partie C", pd.DataFrame(rows))
    _print_frame("attention C2", attention_frame(report.attention))
    print(f"meilleur C1–C3 : {report.best_model}")

    if args.limit is None:
        saved = append_results(rows, ROOT / "results" / "results.csv")
        _print_frame("results/results.csv", saved)
        print("Prédictions de test non écrites : elles sont réservées à la partie F.")
    else:
        print("Prédictions de test non écrites. Journal inchangé à cause de --limit.")


if __name__ == "__main__":
    main()
