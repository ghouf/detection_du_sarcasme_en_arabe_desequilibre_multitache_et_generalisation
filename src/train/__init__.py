"""Training helpers.

Part A fits the TF-IDF baseline. Part C (shared budget, early stopping, imbalance)
lives in ``train.deep`` so importing ``set_seed`` does not load the networks.
"""

from __future__ import annotations

import random
import time

import numpy as np

from models.baseline import build_baseline, n_trainable_params

DEFAULT_SEED = 42


def set_seed(seed: int = DEFAULT_SEED) -> None:
    """Fix random, NumPy, torch, and transformers when those libraries import."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    try:
        from transformers import set_seed as hf_set_seed

        hf_set_seed(seed)
    except ImportError:
        pass


def fit_baseline(tweets, labels, seed: int = DEFAULT_SEED):
    """Fit the baseline on train tweets. Returns the pipeline, seconds, and parameter count."""
    set_seed(seed)
    model = build_baseline()
    started = time.perf_counter()
    model.fit(list(tweets), np.asarray(labels))
    elapsed = time.perf_counter() - started
    return model, elapsed, n_trainable_params(model)
