"""Word vocabulary and AraVec initialization for Part C.

Sequences are tokenized on the lightly normalized tweet. AraVec's ``clean_str``
is applied only when a token is looked up in the AraVec vocabulary, so
elongation, laughs, emoji, and punctuation stay visible to the network.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from preprocess import preprocess_for_tfidf

PAD = "<pad>"
UNK = "<unk>"
PAD_ID = 0
UNK_ID = 1
MIN_FREQ = 2
ARAVEC_DIM = 100

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARAVEC_DIR = PROJECT_ROOT / "data" / "aravec"

_TASHKEEL_RE = re.compile(r"[\u0617-\u061A\u064B-\u0652]")
_LONGATION_RE = re.compile(r"(.)\1+")
_EMOJI_RE: re.Pattern[str] | None = None

# AraVec utilities.clean_str substitutions (bakrianoo/aravec).
_ARAVEC_SEARCH = [
    "أ",
    "إ",
    "آ",
    "ة",
    "_",
    "-",
    "/",
    ".",
    "،",
    " و ",
    " يا ",
    '"',
    "ـ",
    "'",
    "ى",
    "\\",
    "\n",
    "\t",
    "&quot;",
    "?",
    "؟",
    "!",
]
_ARAVEC_REPLACE = [
    "ا",
    "ا",
    "ا",
    "ه",
    " ",
    " ",
    "",
    "",
    "",
    " و",
    " يا",
    "",
    "",
    "",
    "ي",
    "",
    " ",
    " ",
    " ",
    " ? ",
    " ؟ ",
    " ! ",
]


class Vocab:
    """Token-to-id map. Index 0 is padding and index 1 is the unknown token."""

    def __init__(self, itos: list[str]):
        self.itos = list(itos)
        self.stoi = {token: index for index, token in enumerate(self.itos)}

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, tokens: list[str]) -> list[int]:
        return [self.stoi.get(token, UNK_ID) for token in tokens]


class AraVecVectors:
    """Gensim keyed vectors plus the file they were read from."""

    def __init__(self, keyed, path: Path):
        self.keyed = keyed
        self.path = Path(path)
        self.dim = int(keyed.vector_size)

    def __contains__(self, key: str) -> bool:
        return key in self.keyed

    def __getitem__(self, key: str) -> np.ndarray:
        return np.asarray(self.keyed[key], dtype=np.float32)


def clean_str(text: str) -> str:
    """AraVec's ``clean_str``. Lookup only: do not tokenize the model input with it."""
    cleaned = _TASHKEEL_RE.sub("", str(text))
    cleaned = _LONGATION_RE.sub(r"\1\1", cleaned)
    cleaned = cleaned.replace("وو", "و").replace("يي", "ي").replace("اا", "ا")
    for source, target in zip(_ARAVEC_SEARCH, _ARAVEC_REPLACE):
        cleaned = cleaned.replace(source, target)
    return cleaned.strip()


def _emoji_pattern() -> re.Pattern[str]:
    global _EMOJI_RE
    if _EMOJI_RE is None:
        import emoji

        _EMOJI_RE = emoji.get_emoji_regexp()
    return _EMOJI_RE


def _is_word_char(char: str) -> bool:
    """Letters, diacritics, and digits stay inside a word token."""
    return unicodedata.category(char)[0] in {"L", "M", "N"}


def tokenize(text: str) -> list[str]:
    """Words, punctuation runs, and consecutive emojis on the light normalization.

    CSV quote wraps are removed. Sarcasm cues are kept. ``clean_str`` is not applied.
    """
    value = preprocess_for_tfidf(text)
    if not value:
        return []
    emoji_re = _emoji_pattern()
    tokens: list[str] = []
    index = 0
    length = len(value)
    while index < length:
        if value[index].isspace():
            index += 1
            continue
        emoji_match = emoji_re.match(value, index)
        if emoji_match:
            end = emoji_match.end()
            while True:
                nxt = emoji_re.match(value, end)
                if nxt is None:
                    break
                end = nxt.end()
            tokens.append(value[index:end])
            index = end
            continue
        if _is_word_char(value[index]):
            end = index + 1
            while end < length and _is_word_char(value[end]):
                end += 1
            tokens.append(value[index:end])
            index = end
            continue
        end = index + 1
        while (
            end < length
            and not value[end].isspace()
            and not _is_word_char(value[end])
            and emoji_re.match(value, end) is None
        ):
            end += 1
        tokens.append(value[index:end])
        index = end
    return tokens


def build_vocab(texts, min_freq: int = MIN_FREQ) -> Vocab:
    """Vocabulary of the training texts. A type is kept when its count is at least ``min_freq``."""
    counts: Counter[str] = Counter()
    for text in texts:
        counts.update(tokenize(text))
    kept = [token for token, count in counts.items() if count >= min_freq]
    kept.sort(key=lambda token: (-counts[token], token))
    return Vocab([PAD, UNK, *kept])


def build_char_vocab(texts, min_freq: int = 1) -> Vocab:
    """Character vocabulary of the training texts, after the same light normalization."""
    counts: Counter[str] = Counter()
    for text in texts:
        counts.update(preprocess_for_tfidf(text))
    kept = [char for char, count in counts.items() if count >= min_freq]
    kept.sort(key=lambda char: (-counts[char], char))
    return Vocab([PAD, UNK, *kept])


def ids_for_text(text: str, vocab: Vocab, max_len: int) -> list[int]:
    """Encode one tweet. An empty tweet becomes a single unknown id."""
    ids = vocab.encode(tokenize(text))[:max_len]
    if not ids:
        return [UNK_ID]
    return ids


def char_ids_for_text(text: str, vocab: Vocab, max_len: int) -> list[int]:
    chars = list(preprocess_for_tfidf(text))[:max_len]
    ids = [vocab.stoi.get(char, UNK_ID) for char in chars]
    if not ids:
        return [UNK_ID]
    return ids


def oov_stats(texts, vocab: Vocab) -> dict[str, float | int]:
    """Token out-of-vocabulary rate against a vocabulary already built on train."""
    n_tokens = 0
    n_oov = 0
    for text in texts:
        tokens = tokenize(text)
        n_tokens += len(tokens)
        n_oov += sum(token not in vocab.stoi for token in tokens)
    rate = float(n_oov) / n_tokens if n_tokens else float("nan")
    return {"n_tokens": n_tokens, "n_oov": n_oov, "oov_rate": rate}


def oov_table(splits: dict[str, object], vocab: Vocab) -> pd.DataFrame:
    """One row per split. The test row is descriptive: it is not a score."""
    rows = []
    for name, texts in splits.items():
        stats = oov_stats(texts, vocab)
        rows.append({"split": name, **stats})
    return pd.DataFrame(rows)


def aravec_token_coverage(texts, vectors) -> dict[str, float | int]:
    """Share of tokens whose ``clean_str`` form exists in the AraVec table."""
    n_tokens = 0
    n_hits = 0
    for text in texts:
        for token in tokenize(text):
            n_tokens += 1
            key = clean_str(token)
            if key and key in vectors:
                n_hits += 1
    rate = float(n_hits) / n_tokens if n_tokens else float("nan")
    return {"n_tokens": n_tokens, "n_hits": n_hits, "hit_rate": rate}


def find_aravec(dim: int = ARAVEC_DIM, directory: Path | None = None) -> Path | None:
    """Local AraVec file, if the user has placed it under ``data/aravec``."""
    root = Path(directory) if directory is not None else ARAVEC_DIR
    name = f"full_uni_cbow_{dim}_twitter"
    if not root.exists():
        return None
    candidates = [
        root / f"{name}.mdl",
        root / name / f"{name}.mdl",
        root / f"{name}.bin",
        root / name / f"{name}.bin",
    ]
    for path in candidates:
        if path.is_file():
            return path
    matches = sorted(root.rglob(f"{name}.mdl"))
    return matches[0] if matches else None


def load_aravec(dim: int = ARAVEC_DIM, directory: Path | None = None) -> AraVecVectors | None:
    """Load Twitter unigram CBOW vectors. ``None`` when the file is absent.

    Nothing is downloaded. Old AraVec ``.mdl`` files are gensim Word2Vec models;
    a KeyedVectors file is accepted as a fallback.
    """
    path = find_aravec(dim, directory)
    if path is None:
        return None
    from gensim.models import KeyedVectors, Word2Vec

    try:
        loaded = Word2Vec.load(str(path))
    except Exception:
        loaded = KeyedVectors.load(str(path))
    keyed = getattr(loaded, "wv", loaded)
    return AraVecVectors(keyed, path)


def random_embedding_matrix(n_rows: int, dim: int, seed: int) -> np.ndarray:
    """Gaussian rows, with the padding row fixed at zero."""
    rng = np.random.default_rng(seed)
    matrix = rng.normal(0.0, 0.1, size=(n_rows, dim)).astype(np.float32)
    matrix[PAD_ID] = 0.0
    return matrix


def build_embedding_matrix(vocab: Vocab, vectors, dim: int, seed: int) -> tuple[np.ndarray, dict[str, float | int]]:
    """Copy AraVec rows after ``clean_str``. Missing types keep a random vector."""
    if int(vectors.dim) != int(dim):
        raise ValueError(f"AraVec dimension is {vectors.dim}, expected {dim}")
    matrix = random_embedding_matrix(len(vocab), dim, seed)
    hits = 0
    looked = 0
    for token, index in vocab.stoi.items():
        if index <= UNK_ID:
            continue
        looked += 1
        key = clean_str(token)
        if key and key in vectors:
            matrix[index] = np.asarray(vectors[key], dtype=np.float32)
            hits += 1
    coverage = {
        "vocab_types": looked,
        "vocab_hits": hits,
        "vocab_hit_rate": (hits / looked) if looked else float("nan"),
    }
    return matrix, coverage
