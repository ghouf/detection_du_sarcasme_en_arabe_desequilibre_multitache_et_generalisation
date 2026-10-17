"""Multi-task and Mono-task PLM architectures for Partie E."""

from __future__ import annotations

import torch
from torch import nn
from transformers import AutoModel


class MonoTask(nn.Module):
    """E1: Mono-task Sarcasm Detection model using a shared PLM backbone."""

    def __init__(self, name: str, dropout: float = 0.2):
        super().__init__()
        self.enc = AutoModel.from_pretrained(name)
        h = self.enc.config.hidden_size
        self.drop = nn.Dropout(dropout)
        self.sarc = nn.Linear(h, 2)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        out = self.enc(input_ids=input_ids, attention_mask=attention_mask)
        # Representation of the [CLS] token at index 0
        cls = self.drop(out.last_hidden_state[:, 0])
        return self.sarc(cls)


class MultiTask(nn.Module):
    """E2: Multi-task model with shared PLM backbone and 2 heads (sarcasm, sentiment).

    Optionally includes a 3rd head for dialect classification.
    """

    def __init__(
        self,
        name: str,
        num_sarc: int = 2,
        num_sent: int = 3,
        num_dial: int = 5,
        use_dialect: bool = False,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.enc = AutoModel.from_pretrained(name)
        h = self.enc.config.hidden_size
        self.drop = nn.Dropout(dropout)
        self.sarc = nn.Linear(h, num_sarc)
        self.sent = nn.Linear(h, num_sent)
        self.use_dialect = use_dialect
        if use_dialect:
            self.dial = nn.Linear(h, num_dial)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.enc(input_ids=input_ids, attention_mask=attention_mask)
        cls = self.drop(out.last_hidden_state[:, 0])
        logits_sarc = self.sarc(cls)
        logits_sent = self.sent(cls)
        if self.use_dialect:
            logits_dial = self.dial(cls)
            return logits_sarc, logits_sent, logits_dial
        return logits_sarc, logits_sent
