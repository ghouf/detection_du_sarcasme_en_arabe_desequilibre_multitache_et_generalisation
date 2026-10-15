"""Sarcastic F1, the validation threshold, and the prediction file schema."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evaluate import (
    build_part_a_rows,
    choose_threshold,
    circular_partition,
    export_predictions,
    sarcasm_f1,
    trivial_predict,
)


def test_trivial_sarcastic_f1_is_zero():
    labels = np.array([True, False, False, True])
    pred, proba = trivial_predict(len(labels))

    assert sarcasm_f1(labels, pred) == 0.0
    assert proba.sum() == 0.0
    assert pred.sum() == 0


def test_choose_threshold_maximizes_sarcastic_f1():
    labels = np.array([True, True, False, False, False])
    proba = np.array([0.9, 0.5, 0.2, 0.1, 0.0])

    threshold, best_f1 = choose_threshold(labels, proba)
    grid = np.round(np.arange(0.01, 1.00, 0.01), 2)
    grid_f1 = [sarcasm_f1(labels, proba >= value) for value in grid]

    assert best_f1 == max(grid_f1)
    assert sarcasm_f1(labels, proba >= threshold) == best_f1
    assert best_f1 == 1.0


def test_part_a_rows_leave_the_test_empty():
    labels = np.array([True, False, False])
    pred = np.array([True, False, False])
    proba = np.array([0.8, 0.2, 0.1])
    rows = build_part_a_rows(labels, pred, proba, 0.4, 1.2, 10, [], seed=42)

    assert rows[0]["model"] == "trivial_never_sarcastic"
    assert rows[0]["val_accuracy"] != ""
    assert rows[1]["val_accuracy"] == ""
    assert rows[1]["test_f1_sarcasm"] == ""
    assert rows[1]["test_pr_auc"] == ""
    assert rows[1]["test_macro_f1"] == ""


def test_circular_partition_one_pie_per_split():
    splits = {
        "train": pd.DataFrame(
            {
                "sentiment": ["NEG", "NEG", "POS", "NEU"],
                "dialect": ["msa", "egypt", "msa", "gulf"],
            }
        ),
        "val": pd.DataFrame({"sentiment": ["NEU"], "dialect": ["levant"]}),
    }
    figure = circular_partition(
        splits,
        "sentiment",
        "Répartition du sentiment",
        order=["NEG", "NEU", "POS"],
    )
    assert len(figure.axes) == 2
    plt.close(figure)


def test_export_predictions_columns(tmp_path):
    path = tmp_path / "tfidf_logreg_seed42.csv"
    export_predictions(
        path,
        ids=[3, 1],
        y_true=[True, False],
        y_pred=[True, False],
        proba=[0.7, 0.2],
    )
    saved = pd.read_csv(path)
    assert saved.columns.tolist() == ["id", "y_true", "y_pred", "proba_sarcasm"]
    assert saved["id"].tolist() == [3, 1]
