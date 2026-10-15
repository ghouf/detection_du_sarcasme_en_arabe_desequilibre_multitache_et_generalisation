"""Part B helpers that do not download model weights."""

import numpy as np
import pandas as pd
import pytest

from evaluate import SENTIMENT_NOTE, build_part_b_rows
from models.plm import (
    HYPOTHESES,
    group_sentiment_scores,
    hypothesis_frame,
    negative_class,
    sarcasm_from_sentiment_scores,
    sarcastic_score,
    slice_validation,
)


def test_hypotheses_cover_three_languages():
    languages = [item.language for item in HYPOTHESES]
    assert languages == ["ar", "fr", "en"]
    frame = hypothesis_frame()
    assert list(frame["model"]) == ["nli_zeroshot_ar", "nli_zeroshot_fr", "nli_zeroshot_en"]
    for item in HYPOTHESES:
        assert "{}" in item.template
        assert item.positive in item.labels
        assert item.template.format(item.positive)


def test_sarcastic_score_ignores_sort_order():
    output = {"labels": ["جادة", "ساخرة"], "scores": [0.8, 0.2]}
    assert sarcastic_score(output, "ساخرة") == 0.2
    with pytest.raises(ValueError):
        sarcastic_score(output, "sarcastique")


def test_negative_class_is_found_by_name():
    assert negative_class({0: "positive", 1: "negative", 2: "neutral"}) == (1, "negative")
    assert negative_class({"0": "NEU", "2": "NEG"}) == (2, "NEG")
    with pytest.raises(ValueError):
        negative_class({0: "positive", 1: "neutral"})


def test_sentiment_rule_uses_argmax_not_a_threshold():
    losing = [
        {"label": "neutral", "score": 0.5},
        {"label": "negative", "score": 0.4},
        {"label": "positive", "score": 0.1},
    ]
    pred, proba = sarcasm_from_sentiment_scores(losing, "negative")
    assert pred is False
    assert proba == 0.4

    winning = [
        {"label": "NEG", "score": 0.7},
        {"label": "NEU", "score": 0.2},
        {"label": "POS", "score": 0.1},
    ]
    pred, proba = sarcasm_from_sentiment_scores(winning, "NEG")
    assert pred is True
    assert proba == 0.7
    assert sarcasm_from_sentiment_scores(
        [{"label": "POS", "score": 0.9}, {"label": "NEG", "score": 0.1}],
        "NEG",
    )[0] is False


def test_group_sentiment_scores_accepts_single_and_batch():
    single = [{"label": "negative", "score": 0.6}, {"label": "positive", "score": 0.4}]
    batch = [single, [{"label": "neutral", "score": 1.0}]]
    assert group_sentiment_scores(single) == [single]
    assert group_sentiment_scores(batch) == batch
    assert group_sentiment_scores([]) == []


def test_slice_validation_requires_both_labels():
    frame = pd.DataFrame({"tweet": ["a", "b", "c"], "sarcasm": [False, False, True]})
    assert len(slice_validation(frame, None)) == 3
    assert len(slice_validation(frame, 3)) == 3
    with pytest.raises(ValueError):
        slice_validation(frame, 2)
    with pytest.raises(ValueError):
        slice_validation(frame, 0)


def test_part_b_rows_leave_the_test_empty():
    labels = np.array([True, False, False])
    pred = np.array([True, False, True])
    proba = np.array([0.8, 0.2, 0.6])
    rows = build_part_b_rows(
        labels,
        [
            ("nli_zeroshot_ar", pred, proba, 0.4, 1.5),
            ("nli_zeroshot_fr", pred, proba, 0.5, 1.6),
            ("nli_zeroshot_en", pred, proba, 0.3, 1.7),
        ],
        pred,
        proba,
        0.4,
        seed=42,
    )

    assert [row["model"] for row in rows] == [
        "nli_zeroshot_ar",
        "nli_zeroshot_fr",
        "nli_zeroshot_en",
        "sentiment_neg_implies_sarcasm",
    ]
    assert all(row["preprocess"] == "raw" for row in rows)
    assert all(row["n_trainable_params"] == 0 for row in rows)
    assert all(row["test_f1_sarcasm"] == "" for row in rows)
    assert all(row["test_pr_auc"] == "" for row in rows)
    assert all(row["test_macro_f1"] == "" for row in rows)
    assert rows[0]["threshold"] == 0.4
    assert rows[3]["threshold"] == ""
    assert "ASTD" in rows[3]["notes"]
    assert rows[3]["notes"] == SENTIMENT_NOTE
