"""LSTM / Bi-LSTM sentiment classifier.

A single class covers both architectures: ``RNNConfig.bidirectional`` is the only
switch. Many-to-one classification uses the final hidden state of the last layer
(forward, or forward+backward concatenated for the bidirectional variant).

Sequences are packed using their true lengths so trailing pad tokens never
contribute to the final hidden state.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..config import RNNConfig
from ..vocab import PAD_ID


class LSTMClassifier(nn.Module):
    def __init__(self, cfg: RNNConfig, vocab_size: int, num_classes: int):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(vocab_size, cfg.embed_dim, padding_idx=PAD_ID)
        self.lstm = nn.LSTM(
            input_size=cfg.embed_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            batch_first=True,
            bidirectional=cfg.bidirectional,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(cfg.dropout)
        directions = 2 if cfg.bidirectional else 1
        self.fc = nn.Linear(cfg.hidden_dim * directions, num_classes)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # input_ids: (B, L)  lengths: (B,)
        embedded = self.embedding(input_ids)  # (B, L, E)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)  # h_n: (num_layers*dirs, B, H)

        if self.cfg.bidirectional:
            # Last layer's forward (-2) and backward (-1) hidden states.
            last = torch.cat([h_n[-2], h_n[-1]], dim=1)  # (B, 2H)
        else:
            last = h_n[-1]  # (B, H)

        return self.fc(self.dropout(last))  # (B, num_classes)
