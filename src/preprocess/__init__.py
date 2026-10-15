"""Raw-text audit and the two text views used by Part A.

The audit runs on the original string. Dedup normalization and the TF-IDF
string are separate: stripping sarcasm cues (emoji, laughs, elongation,
punctuation) is not part of the baseline.
"""

from __future__ import annotations

import re

import emoji
import pandas as pd
from pyarabic.araby import DIACRITICS, TATWEEL

_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_MENTION_RE = re.compile(r"(?<!\w)@\w+")
_HASHTAG_RE = re.compile(r"(?<!\w)#\w+")
_LAUGH_RE = re.compile(r"ه{2,}")
_REPEATED_PUNCT_RE = re.compile(r"([!?؟.,،؛…])\1+")
_REPEATED_LETTER_RE = re.compile(r"([^\W\d_])\1{2,}")
_LATIN_RE = re.compile(r"[A-Za-z]")
_INDIAN_DIGIT_RE = re.compile(r"[٠-٩]")
_WS_RE = re.compile(r"\s+")

_DIACRITIC_CHARS = "".join(ch for ch in DIACRITICS if ch != TATWEEL)
_DIACRITIC_RE = re.compile("[" + re.escape(_DIACRITIC_CHARS) + "]")

SURFACE_FEATURES: tuple[tuple[str, str, re.Pattern[str] | None], ...] = (
    ("has_url", "url", _URL_RE),
    ("has_mention", "mention", _MENTION_RE),
    ("has_hashtag", "hashtag", _HASHTAG_RE),
    ("has_emoji", "emoji", None),
    ("has_laugh", "laugh_heh", _LAUGH_RE),
    ("has_repeated_punct", "repeated_punctuation", _REPEATED_PUNCT_RE),
    ("has_diacritics", "diacritics", _DIACRITIC_RE),
    ("has_tatweel", "tatweel", re.compile(re.escape(TATWEEL))),
    ("has_repeated_letters", "repeated_letters", _REPEATED_LETTER_RE),
    ("has_latin", "latin", _LATIN_RE),
    ("has_indian_digits", "indian_digits", _INDIAN_DIGIT_RE),
    ("has_csv_quote", "csv_quote_wrap", None),
)


def has_csv_quote_wrap(text: str) -> bool:
    """True when the whole tweet is wrapped in ASCII double quotes.

    Those quotes come from the CSV export. Curly quotes inside a tweet are
    not this artifact.
    """
    stripped = str(text).strip()
    return len(stripped) >= 2 and stripped[0] == '"' and stripped[-1] == '"'


def strip_csv_quotes(text: str) -> str:
    stripped = str(text).strip()
    if len(stripped) >= 2 and stripped[0] == '"' and stripped[-1] == '"':
        return stripped[1:-1].strip()
    return stripped


def normalize_for_dedup(text: str) -> str:
    """Key for duplicate detection only. Not the TF-IDF input."""
    cleaned = strip_csv_quotes(text).replace(TATWEEL, "")
    return _WS_RE.sub(" ", cleaned).strip()


def preprocess_for_tfidf(text: str) -> str:
    """Light cleaner: drop the CSV quote wrap, keep sarcasm cues."""
    return _WS_RE.sub(" ", strip_csv_quotes(text)).strip()


def emoji_occurrences(text: str) -> int:
    """Count emojis with whichever emoji API is installed.

    arabert pins emoji==1.4.2, whose lister is ``emoji_lis``.
    """
    value = str(text)
    if hasattr(emoji, "emoji_count"):
        return int(emoji.emoji_count(value))
    lister = getattr(emoji, "emoji_list", None) or emoji.emoji_lis
    return len(lister(value))


def _presence(tweets: pd.Series, column: str, pattern: re.Pattern[str] | None) -> pd.Series:
    if column == "has_csv_quote":
        return tweets.map(has_csv_quote_wrap)
    if column == "has_emoji":
        return tweets.map(lambda text: emoji_occurrences(text) > 0)
    # Python re, not Series.str: pandas 3 / Arrow rejects backreferences such as \1.
    return tweets.map(lambda text: pattern.search(str(text)) is not None)


def _occurrences(tweets: pd.Series, column: str, pattern: re.Pattern[str] | None) -> int:
    if column == "has_csv_quote":
        return int(tweets.map(has_csv_quote_wrap).sum())
    if column == "has_emoji":
        return int(tweets.map(emoji_occurrences).sum())
    return sum(len(pattern.findall(str(text))) for text in tweets)


def add_surface_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add boolean presence columns computed on the raw tweet."""
    out = df.copy()
    tweets = out["tweet"].fillna("").astype(str)
    for column, _label, pattern in SURFACE_FEATURES:
        out[column] = _presence(tweets, column, pattern)
    return out


def audit_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Frequency of each surface cue and its link with the sarcasm label."""
    tweets = df["tweet"].fillna("").astype(str)
    labels = df["sarcasm"].astype(bool)
    rows: list[dict[str, object]] = []
    for column, label, pattern in SURFACE_FEATURES:
        present = _presence(tweets, column, pattern).astype(bool)
        present_labels = labels[present]
        absent_labels = labels[~present]
        correlation = present.astype(float).corr(labels.astype(float))
        rows.append(
            {
                "feature": label,
                "n_tweets": int(present.sum()),
                "frequency": float(present.mean()) if len(present) else float("nan"),
                "n_occurrences": _occurrences(tweets, column, pattern),
                "sarcasm_rate_present": float(present_labels.mean()) if present.any() else float("nan"),
                "sarcasm_rate_absent": float(absent_labels.mean()) if (~present).any() else float("nan"),
                "pearson_with_sarcasm": float(correlation) if pd.notna(correlation) else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def _duplicate_stats(texts: pd.Series) -> dict[str, int]:
    values = texts.astype(str)
    in_group = values.duplicated(keep=False)
    return {
        "rows_in_duplicate_groups": int(in_group.sum()),
        "extra_copies": int(values.duplicated(keep="first").sum()),
        "groups": int(values[in_group].nunique()) if in_group.any() else 0,
    }


def contradictory_label_stats(df: pd.DataFrame) -> dict[str, int]:
    """Groups whose normalized text occurs with both sarcasm labels."""
    keys = df["tweet"].map(normalize_for_dedup)
    nunique = df.assign(_key=keys).groupby("_key")["sarcasm"].nunique()
    bad_keys = set(nunique[nunique > 1].index)
    return {
        "groups": len(bad_keys),
        "rows": int(keys.isin(bad_keys).sum()),
    }


def duplicate_report(left: pd.DataFrame, right: pd.DataFrame, left_name: str = "train", right_name: str = "test") -> dict[str, int | str]:
    """Exact and normalized duplicates inside ``left``, and overlap with ``right``."""
    left_text = left["tweet"].astype(str)
    right_text = right["tweet"].astype(str)
    left_norm = left_text.map(normalize_for_dedup)
    right_norm = set(right_text.map(normalize_for_dedup))
    exact = _duplicate_stats(left_text)
    normalized = _duplicate_stats(left_norm)
    contradictory = contradictory_label_stats(left)
    return {
        "left": left_name,
        "right": right_name,
        "exact_rows_in_duplicate_groups": exact["rows_in_duplicate_groups"],
        "exact_extra_copies": exact["extra_copies"],
        "exact_groups": exact["groups"],
        "normalized_rows_in_duplicate_groups": normalized["rows_in_duplicate_groups"],
        "normalized_extra_copies": normalized["extra_copies"],
        "normalized_groups": normalized["groups"],
        "exact_overlap_rows": int(left_text.isin(set(right_text)).sum()),
        "normalized_overlap_rows": int(left_norm.isin(right_norm).sum()),
        "contradictory_groups": contradictory["groups"],
        "contradictory_rows": contradictory["rows"],
    }
