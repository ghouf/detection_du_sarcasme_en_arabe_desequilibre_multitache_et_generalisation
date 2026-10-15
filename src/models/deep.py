"""Part C networks: BiLSTM, additive attention, a mini Transformer, and a char-CNN.

Every classifier consumes padded token ids and a boolean mask (True = real token)
and returns sarcasm logits of shape (batch, 2). Class 1 is sarcastic.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from models.static import PAD_ID

D_MODEL = 128
N_HEADS = 4
N_LAYERS = 2
CHAR_EMBED_DIM = 32
CHAR_CHANNELS = 64
CHAR_KERNELS = (3, 5, 7)


def sinusoidal_position_encoding(max_len: int, d_model: int) -> torch.Tensor:
    """Vaswani sinusoidal encoding, shape (max_len, d_model)."""
    if d_model < 1 or max_len < 1:
        raise ValueError("positional encoding needs a positive length and width")
    position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
    )
    encoding = torch.zeros(max_len, d_model, dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(position * div)
    encoding[:, 1::2] = torch.cos(position * div[: encoding[:, 1::2].size(1)])
    return encoding


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        encoding = sinusoidal_position_encoding(max_len, d_model).unsqueeze(0)
        self.register_buffer("pe", encoding, persistent=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        length = hidden.size(1)
        if length > self.pe.size(1):
            raise ValueError(
                f"sequence length {length} exceeds positional encoding {self.pe.size(1)}"
            )
        return hidden + self.pe[:, :length]


def masked_softmax(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Softmax over the time axis. Padding positions (mask False) get weight 0."""
    filled = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
    weights = torch.softmax(filled, dim=-1)
    weights = weights.masked_fill(~mask, 0.0)
    return torch.nan_to_num(weights, nan=0.0)


class AdditiveAttention(nn.Module):
    """αᵢ = softmax(vᵀ tanh(W hᵢ)), then c = Σ αᵢ hᵢ."""

    def __init__(self, hidden: int):
        super().__init__()
        self.W = nn.Linear(hidden, hidden, bias=False)
        self.v = nn.Linear(hidden, 1, bias=False)

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self.v(torch.tanh(self.W(hidden))).squeeze(-1)
        alpha = masked_softmax(scores, mask)
        context = torch.bmm(alpha.unsqueeze(1), hidden).squeeze(1)
        return context, alpha


def _embedding_from_matrix(matrix: np.ndarray, freeze: bool) -> nn.Embedding:
    weight = torch.tensor(np.array(matrix), dtype=torch.float32)
    return nn.Embedding.from_pretrained(weight, freeze=freeze, padding_idx=PAD_ID)


class BiLSTMClassifier(nn.Module):
    """C1. Bidirectional LSTM with max-pooling or the last hidden state."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden: int,
        dropout: float = 0.3,
        pool: str = "max",
        embedding: nn.Embedding | None = None,
    ):
        super().__init__()
        if pool not in {"max", "last"}:
            raise ValueError(f"pool must be 'max' or 'last', got {pool!r}")
        self.pool = pool
        self.embedding = embedding or nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_ID)
        self.lstm = nn.LSTM(embed_dim, hidden, num_layers=1, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden * 2, 2)

    def _encode(self, ids: torch.Tensor, mask: torch.Tensor):
        embedded = self.dropout(self.embedding(ids))
        lengths = mask.sum(dim=1).clamp(min=1).detach().cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths, batch_first=True, enforce_sorted=False
        )
        packed_out, (h_n, _cell) = self.lstm(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True, total_length=ids.size(1)
        )
        return output, h_n

    def _pool(self, states: torch.Tensor, h_n: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.pool == "last":
            # h_n is (2, batch, hidden): forward final state, then backward final state.
            return torch.cat([h_n[0], h_n[1]], dim=-1)
        filled = states.masked_fill(~mask.unsqueeze(-1), torch.finfo(states.dtype).min)
        pooled = filled.max(dim=1).values
        return torch.nan_to_num(pooled, nan=0.0, neginf=0.0)

    def forward(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        states, h_n = self._encode(ids, mask)
        pooled = self._pool(states, h_n, mask)
        return self.fc(self.dropout(pooled))


class BiLSTMAttention(nn.Module):
    """C2. The same BiLSTM, pooled by the additive attention above."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden: int,
        dropout: float = 0.3,
        embedding: nn.Embedding | None = None,
    ):
        super().__init__()
        self.embedding = embedding or nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_ID)
        self.lstm = nn.LSTM(embed_dim, hidden, num_layers=1, batch_first=True, bidirectional=True)
        self.attention = AdditiveAttention(hidden * 2)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden * 2, 2)

    def _states(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        embedded = self.dropout(self.embedding(ids))
        lengths = mask.sum(dim=1).clamp(min=1).detach().cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths, batch_first=True, enforce_sorted=False
        )
        packed_out, _state = self.lstm(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True, total_length=ids.size(1)
        )
        return output

    def forward(
        self,
        ids: torch.Tensor,
        mask: torch.Tensor,
        return_attention: bool = False,
    ):
        states = self._states(ids, mask)
        context, alpha = self.attention(states, mask)
        logits = self.fc(self.dropout(context))
        if return_attention:
            return logits, alpha
        return logits


class MiniTransformer(nn.Module):
    """C3. Two encoder layers, d_model 128, 4 heads, optional sinusoidal positions.

    Padding is ignored via ``src_key_padding_mask``. The sentence vector is the
    mean of the non-padding positions.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        d_model: int = D_MODEL,
        nhead: int = N_HEADS,
        num_layers: int = N_LAYERS,
        dropout: float = 0.3,
        use_positional: bool = True,
        max_len: int = 512,
        embedding: nn.Embedding | None = None,
    ):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model {d_model} must be divisible by nhead {nhead}")
        self.embedding = embedding or nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_ID)
        self.project = nn.Identity() if embed_dim == d_model else nn.Linear(embed_dim, d_model)
        self.positional = SinusoidalPositionalEncoding(d_model, max_len) if use_positional else None
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers, enable_nested_tensor=False)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, 2)

    def forward(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        hidden = self.project(self.embedding(ids))
        if self.positional is not None:
            hidden = self.positional(hidden)
        hidden = self.dropout(hidden)
        # True marks positions the encoder must ignore.
        encoded = self.encoder(hidden, src_key_padding_mask=~mask)
        weights = mask.to(dtype=encoded.dtype).unsqueeze(-1)
        pooled = (encoded * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)
        return self.fc(self.dropout(pooled))


class CharCNN(nn.Module):
    """C4. Character CNN with kernels 3, 5, and 7. Optional sarcasm-cue probe."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = CHAR_EMBED_DIM,
        channels: int = CHAR_CHANNELS,
        kernels: tuple[int, ...] = CHAR_KERNELS,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_ID)
        self.convs = nn.ModuleList(
            [nn.Conv1d(embed_dim, channels, kernel, padding=kernel // 2) for kernel in kernels]
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(channels * len(kernels), 2)

    def forward(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        hidden = self.embedding(ids).transpose(1, 2)
        pooled = []
        keep = mask.unsqueeze(1)
        for conv in self.convs:
            activated = torch.relu(conv(hidden))
            filled = activated.masked_fill(~keep, torch.finfo(activated.dtype).min)
            pooled.append(torch.nan_to_num(filled.max(dim=-1).values, nan=0.0, neginf=0.0))
        return self.fc(self.dropout(torch.cat(pooled, dim=-1)))


def build_classifier(
    arch: str,
    vocab_size: int,
    embed_dim: int,
    *,
    hidden: int,
    dropout: float,
    d_model: int,
    nhead: int,
    num_layers: int,
    pool: str,
    max_len: int,
    embedding: nn.Embedding | None = None,
) -> nn.Module:
    """Build one Part C architecture. ``embedding`` is shared init for word models."""
    if arch == "c1":
        return BiLSTMClassifier(
            vocab_size, embed_dim, hidden, dropout=dropout, pool=pool, embedding=embedding
        )
    if arch == "c2":
        return BiLSTMAttention(
            vocab_size, embed_dim, hidden, dropout=dropout, embedding=embedding
        )
    if arch == "c3":
        return MiniTransformer(
            vocab_size,
            embed_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout,
            use_positional=True,
            max_len=max_len,
            embedding=embedding,
        )
    if arch == "c3_no_pe":
        return MiniTransformer(
            vocab_size,
            embed_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout,
            use_positional=False,
            max_len=max_len,
            embedding=embedding,
        )
    if arch == "c4":
        return CharCNN(vocab_size, dropout=dropout)
    raise ValueError(f"unknown architecture {arch!r}")


def count_trainable(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


def embedding_from_matrix(matrix: np.ndarray, freeze: bool) -> nn.Embedding:
    return _embedding_from_matrix(matrix, freeze)
