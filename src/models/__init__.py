"""Model families for the TP.

Part A implements the TF-IDF logistic-regression baseline. Later parts add
the zero-shot PLM, static embeddings, the frozen encoder, and fine-tuning.
"""

from models.baseline import build_baseline

__all__ = ["build_baseline"]
