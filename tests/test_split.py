"""Imposed split and the train-only leakage filter."""

import pandas as pd
from pyarabic.araby import TATWEEL

from data import drop_contradictory, drop_test_leakage, imposed_split


def _frame(tweets, labels):
    return pd.DataFrame(
        {
            "id": range(len(tweets)),
            "tweet": tweets,
            "sarcasm": labels,
            "sentiment": ["NEG"] * len(tweets),
            "dialect": ["msa"] * len(tweets),
        }
    )


def test_imposed_split_is_stratified_and_reproducible():
    tweets = [f"t{i}" for i in range(100)]
    labels = [True] * 20 + [False] * 80
    full = _frame(tweets, labels)

    train, val = imposed_split(full)

    assert len(val) == 10
    assert len(train) == 90
    assert set(train["id"]).isdisjoint(set(val["id"]))
    assert int(val["sarcasm"].sum()) == 2
    assert int(train["sarcasm"].sum()) == 18

    train_again, val_again = imposed_split(full)
    assert train_again["id"].tolist() == train["id"].tolist()
    assert val_again["id"].tolist() == val["id"].tolist()


def test_test_text_is_removed_from_train_and_kept_in_val():
    train = _frame(
        ["hello", '"hello"', "unique", "نص" + TATWEEL],
        [False, False, False, True],
    )
    val = _frame(["hello", "stay"], [False, True])
    test = _frame(["hello", "نص"], [True, False])

    cleaned, val_out, info = drop_test_leakage(train, val, test)

    assert cleaned["tweet"].tolist() == ["unique"]
    assert "hello" in set(val_out["tweet"])
    assert info["n_train_removed"] == 3
    assert info["n_val_overlap"] == 1

    test_exact = set(test["tweet"])
    test_norm = {"hello", "نص"}
    assert set(cleaned["tweet"]).isdisjoint(test_exact)
    assert set(cleaned["tweet"]).isdisjoint(test_norm)


def test_contradictory_labels_are_dropped_from_train_only():
    train = _frame(
        ["same", "same", "ok", "ok"],
        [True, False, True, True],
    )
    cleaned, stats = drop_contradictory(train)

    assert stats["groups"] == 1
    assert stats["rows"] == 2
    assert cleaned["tweet"].tolist() == ["ok", "ok"]
