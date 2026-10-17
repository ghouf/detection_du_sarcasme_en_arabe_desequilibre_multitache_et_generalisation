"""Model families for the TP.

Part A implements the TF-IDF logistic-regression baseline. Part B uses a
zero-shot NLI model and a sentiment heuristic, with no training. Part C keeps
its word vocabulary in ``models.static`` and its networks in ``models.deep``.
Later parts add the frozen encoder and fine-tuning.
"""

from models.baseline import build_baseline
from models.multitask import MonoTask, MultiTask
from models.plm import (
    HYPOTHESES,
    NLI_MODEL,
    SENTIMENT_MODEL,
    collect_part_b,
    load_nli_pipeline,
    load_sentiment_pipeline,
)

__all__ = [
    "HYPOTHESES",
    "NLI_MODEL",
    "SENTIMENT_MODEL",
    "build_baseline",
    "collect_part_b",
    "load_nli_pipeline",
    "load_sentiment_pipeline",
    "MonoTask",
    "MultiTask",
]
