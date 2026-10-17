
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from preprocess import contradictory_label_stats, normalize_for_dedup

BASE_URL = "https://raw.githubusercontent.com/iabufarha/ArSarcasm-v2/main/ArSarcasm-v2/"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = PROJECT_ROOT / "data" / "raw"
SPLIT_SEED = 42
SPLIT_TEST_SIZE = 0.10


def _as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.upper().isin(["TRUE", "1", "T", "YES"])


def _read_split_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["tweet"] = frame["tweet"].fillna("").astype(str)
    frame["sarcasm"] = _as_bool(frame["sarcasm"])
    frame["sentiment"] = frame["sentiment"].astype(str)
    frame["dialect"] = frame["dialect"].astype(str)
    return frame


def load_raw(cache_dir: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download the official CSVs once, then read the local cache."""
    cache_dir = Path(cache_dir) if cache_dir is not None else CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    for name in ("training_data.csv", "testing_data.csv"):
        path = cache_dir / name
        if not path.exists():
            downloaded = pd.read_csv(BASE_URL + name)
            downloaded.to_csv(path, index=False)
        frames.append(_read_split_csv(path))
    return frames[0], frames[1]


def imposed_split(
    full: pd.DataFrame,
    test_size: float = SPLIT_TEST_SIZE,
    random_state: int = SPLIT_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The split required by the assignment. ``full`` must already carry ``id``."""
    train, val = train_test_split(
        full,
        test_size=test_size,
        random_state=random_state,
        stratify=full["sarcasm"],
    )
    return train.reset_index(drop=True), val.reset_index(drop=True)


def drop_test_leakage(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Drop train rows whose exact or normalized text appears in the official test.

    Validation rows are counted and kept, so the imposed 10% split stays intact.
    """
    test_exact = set(test["tweet"].astype(str))
    test_norm = set(test["tweet"].map(normalize_for_dedup))
    train_exact = train["tweet"].astype(str)
    train_norm = train["tweet"].map(normalize_for_dedup)
    exact_hit = train_exact.isin(test_exact)
    norm_hit = train_norm.isin(test_norm)
    leak_mask = exact_hit | norm_hit

    val_exact = val["tweet"].astype(str)
    val_norm = val["tweet"].map(normalize_for_dedup)
    info = {
        "n_train_exact_overlap": int(exact_hit.sum()),
        "n_train_normalized_overlap": int(norm_hit.sum()),
        "n_train_removed": int(leak_mask.sum()),
        "n_val_overlap": int((val_exact.isin(test_exact) | val_norm.isin(test_norm)).sum()),
    }
    cleaned = train.loc[~leak_mask].reset_index(drop=True)
    return cleaned, val.reset_index(drop=True), info


def drop_contradictory(train: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Drop every train copy of a normalized text that has both labels.

    Same-label duplicates are kept. The official test file is not modified.
    """
    keys = train["tweet"].map(normalize_for_dedup)
    stats = contradictory_label_stats(train)
    if stats["groups"] == 0:
        return train.reset_index(drop=True), stats
    nunique = train.assign(_key=keys).groupby("_key")["sarcasm"].nunique()
    bad_keys = set(nunique[nunique > 1].index)
    kept = train.loc[~keys.isin(bad_keys)].reset_index(drop=True)
    return kept, stats


def load_and_split(cache_dir: Path | None = None) -> dict[str, object]:
    """Official load, imposed split, then train-only leakage and label cleanup."""
    full, test = load_raw(cache_dir)
    full = full.copy()
    test = test.copy()
    full["id"] = range(len(full))
    test["id"] = range(len(test))

    train, val = imposed_split(full)
    train, val, leak = drop_test_leakage(train, val, test)
    train, contradictory = drop_contradictory(train)
    return {
        "full": full,
        "test": test,
        "train": train,
        "val": val,
        "leak": leak,
        "contradictory": contradictory,
    }


def sample_raw_tweets(df: pd.DataFrame, n: int = 30, seed: int = SPLIT_SEED) -> pd.DataFrame:
    """Reproducible raw tweets, before any normalization."""
    columns = ["id", "tweet", "sarcasm", "sentiment", "dialect"]
    taken = min(n, len(df))
    return df.sample(n=taken, random_state=seed)[columns].reset_index(drop=True)
