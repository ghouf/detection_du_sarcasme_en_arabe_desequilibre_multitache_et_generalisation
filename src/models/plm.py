"""Zero-shot NLI and the sentiment heuristic. Neither model is trained here."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

NLI_MODEL = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
SENTIMENT_MODEL = "CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment"
SENTIMENT_ROW_NAME = "sentiment_neg_implies_sarcasm"


@dataclass(frozen=True)
class Hypothesis:
    name: str
    language: str
    template: str
    labels: tuple[str, str]
    positive: str


HYPOTHESES: tuple[Hypothesis, ...] = (
    Hypothesis(
        name="nli_zeroshot_ar",
        language="ar",
        template="هذه التغريدة {}.",
        labels=("ساخرة", "جادة"),
        positive="ساخرة",
    ),
    Hypothesis(
        name="nli_zeroshot_fr",
        language="fr",
        template="Ce tweet est {}.",
        labels=("sarcastique", "sérieux"),
        positive="sarcastique",
    ),
    Hypothesis(
        name="nli_zeroshot_en",
        language="en",
        template="This tweet is {}.",
        labels=("sarcastic", "serious"),
        positive="sarcastic",
    ),
)


@dataclass
class NliRun:
    name: str
    proba: np.ndarray
    pred: np.ndarray
    threshold: float
    elapsed_s: float


@dataclass
class PartBReport:
    nli: list[NliRun]
    sentiment_pred: np.ndarray
    sentiment_proba: np.ndarray
    sentiment_elapsed_s: float
    id2label: dict[int, str]


def hypothesis_frame() -> pd.DataFrame:
    """One row per formulation: template, labels, and the filled sarcastic hypothesis."""
    return pd.DataFrame(
        [
            {
                "model": item.name,
                "langue": item.language,
                "gabarit": item.template,
                "etiquettes": " / ".join(item.labels),
                "positive": item.positive,
                "hypothese": item.template.format(item.positive),
            }
            for item in HYPOTHESES
        ]
    )


def sarcastic_score(output: dict, positive_label: str) -> float:
    """Score of the sarcastic label. The pipeline sorts labels by descending score."""
    labels = list(output["labels"])
    scores = list(output["scores"])
    try:
        index = labels.index(positive_label)
    except ValueError as exc:
        raise ValueError(f"Étiquette sarcastique {positive_label!r} absente de {labels}") from exc
    return float(scores[index])


def negative_class(id2label: dict) -> tuple[int, str]:
    """Index and name of the negative class, matched by name rather than a fixed id."""
    for key, label in id2label.items():
        if "neg" in str(label).lower():
            return int(key), str(label)
    raise ValueError(f"Aucune classe négative dans id2label={id2label}")


def sentiment_is_sarcastic(predicted_label: str, negative_label: str) -> bool:
    return str(predicted_label).casefold() == str(negative_label).casefold()


def negative_probability(scores: list[dict], negative_label: str) -> float:
    """Probability of the negative class inside one text-classification output."""
    for item in scores:
        if str(item["label"]).casefold() == str(negative_label).casefold():
            return float(item["score"])
    labels = [item.get("label") for item in scores]
    raise ValueError(f"Classe {negative_label!r} absente de {labels}")


def sarcasm_from_sentiment_scores(scores: list[dict], negative_label: str) -> tuple[bool, float]:
    """Argmax decision: sarcastic only when the predicted sentiment is the negative class.

    The returned probability is P(negative), even when another class wins.
    """
    top = max(scores, key=lambda item: float(item["score"]))
    return sentiment_is_sarcastic(top["label"], negative_label), negative_probability(scores, negative_label)


def group_sentiment_scores(outputs) -> list[list[dict]]:
    """Normalize a pipeline return into one score list per tweet.

    A single string comes back as a list of dicts. A batch comes back as a list of those lists.
    """
    if not outputs:
        return []
    if isinstance(outputs[0], dict):
        return [list(outputs)]
    return [list(item) for item in outputs]


def slice_validation(val_df: pd.DataFrame, limit: int | None) -> pd.DataFrame:
    """Full validation, or a prefix that still contains both sarcasm labels."""
    if limit is None:
        return val_df.reset_index(drop=True)
    if limit < 1:
        raise ValueError("limit doit être >= 1")
    sliced = val_df.iloc[:limit].reset_index(drop=True)
    if sliced["sarcasm"].nunique() < 2:
        raise ValueError("Ce préfixe de la validation ne contient qu'une classe. Augmentez limit.")
    return sliced


def inference_device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def load_nli_pipeline(device: str | None = None):
    from transformers import pipeline

    if device is None:
        device = inference_device()
    return pipeline("zero-shot-classification", model=NLI_MODEL, device=device)


def load_sentiment_pipeline(device: str | None = None):
    from transformers import pipeline

    if device is None:
        device = inference_device()
    return pipeline("text-classification", model=SENTIMENT_MODEL, device=device)


def _as_texts(tweets) -> list[str]:
    return ["" if tweet is None else str(tweet) for tweet in tweets]


def zero_shot_sarcasm_proba(
    pipe,
    tweets,
    hypothesis: Hypothesis,
    batch_size: int = 8,
) -> tuple[np.ndarray, float]:
    """P(sarcastic) for one hypothesis template. The pipeline already truncates the premise."""
    texts = _as_texts(tweets)
    if not texts:
        return np.zeros(0, dtype=float), 0.0
    started = time.perf_counter()
    outputs = pipe(
        texts,
        candidate_labels=list(hypothesis.labels),
        hypothesis_template=hypothesis.template,
        multi_label=False,
        batch_size=batch_size,
    )
    elapsed = time.perf_counter() - started
    if isinstance(outputs, dict):
        outputs = [outputs]
    proba = np.array([sarcastic_score(item, hypothesis.positive) for item in outputs], dtype=float)
    return proba, elapsed


def sentiment_sarcasm(pipe, tweets, batch_size: int = 8) -> tuple[np.ndarray, np.ndarray, dict[int, str], float]:
    """Negative predicted sentiment means sarcastic. Probability is P(negative)."""
    id2label = {int(key): str(value) for key, value in pipe.model.config.id2label.items()}
    _index, negative_name = negative_class(id2label)
    texts = _as_texts(tweets)
    if not texts:
        return np.zeros(0, dtype=bool), np.zeros(0, dtype=float), id2label, 0.0
    started = time.perf_counter()
    outputs = pipe(texts, top_k=None, truncation=True, batch_size=batch_size)
    elapsed = time.perf_counter() - started
    grouped = group_sentiment_scores(outputs)
    pred = np.empty(len(grouped), dtype=bool)
    proba = np.empty(len(grouped), dtype=float)
    for index, scores in enumerate(grouped):
        pred[index], proba[index] = sarcasm_from_sentiment_scores(scores, negative_name)
    return pred, proba, id2label, elapsed


def collect_part_b(val_df: pd.DataFrame, nli_pipe, sentiment_pipe, batch_size: int = 8) -> PartBReport:
    """Score validation with the three NLI formulations and the sentiment rule.

    The NLI threshold maximizes sarcastic F1 on this frame. The official test is not passed in.
    """
    from evaluate import choose_threshold, predict_at_threshold

    y_val = np.asarray(val_df["sarcasm"]).astype(bool)
    tweets = val_df["tweet"]
    nli_runs: list[NliRun] = []
    for hypothesis in HYPOTHESES:
        proba, elapsed = zero_shot_sarcasm_proba(nli_pipe, tweets, hypothesis, batch_size=batch_size)
        threshold, _best_f1 = choose_threshold(y_val, proba)
        pred = np.asarray(predict_at_threshold(proba, threshold), dtype=bool)
        nli_runs.append(NliRun(hypothesis.name, proba, pred, float(threshold), elapsed))
    pred, proba, id2label, elapsed = sentiment_sarcasm(sentiment_pipe, tweets, batch_size=batch_size)
    return PartBReport(nli_runs, pred, proba, elapsed, id2label)
