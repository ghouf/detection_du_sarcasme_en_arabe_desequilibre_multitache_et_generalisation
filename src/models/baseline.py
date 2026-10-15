"""TF-IDF word (1–2) + character (2–5) logistic regression."""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer

from preprocess import preprocess_for_tfidf


def transform_texts(tweets) -> list[str]:
    return [preprocess_for_tfidf(tweet) for tweet in tweets]


def build_baseline() -> Pipeline:
    """Cleaner, TF-IDF, and classifier. Fit the pipeline on train tweets only."""
    word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2))
    char = TfidfVectorizer(analyzer="char", ngram_range=(2, 5))
    features = FeatureUnion([("word", word), ("char", char)])
    classifier = LogisticRegression(
        class_weight="balanced",
        solver="liblinear",
        max_iter=1000,
        random_state=42,
    )
    cleaner = FunctionTransformer(transform_texts, validate=False)
    return Pipeline([("clean", cleaner), ("tfidf", features), ("clf", classifier)])


def n_trainable_params(model: Pipeline) -> int:
    classifier = model.named_steps["clf"]
    return int(classifier.coef_.size + classifier.intercept_.size)


def sarcastic_proba(model: Pipeline, tweets) -> np.ndarray:
    """Probability of the sarcastic class. Texts are cleaned the same way as in training."""
    proba = model.predict_proba(list(tweets))
    classes = list(model.named_steps["clf"].classes_)
    if True in classes:
        index = classes.index(True)
    elif 1 in classes:
        index = classes.index(1)
    else:
        raise ValueError(f"Sarcastic class not found in {classes}")
    return proba[:, index]
