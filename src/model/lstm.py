"""PyTorch LSTM architecture for dual-head RUL regression + failure classification."""

from __future__ import annotations

import torch
import torch.nn as nn


class RULPredictor(nn.Module):
    """Dual-head LSTM: RUL regression + binary failure-within-30-cycles classification.

    Input shape : (batch, sequence_len=30, n_features=15)
    Outputs     : {"rul": (batch, 1), "failure_prob": (batch, 1)}
    """

    def __init__(
        self,
        n_features: int = 15,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        window_size: int = 30,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.n_features = n_features
        self.window_size = window_size

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)

        # Shared FC layer
        self.fc1 = nn.Linear(hidden_size, 64)
        self.relu = nn.ReLU()

        # RUL regression head
        self.rul_head = nn.Linear(64, 1)

        # Failure classification head (failure within window_size cycles)
        self.failure_head = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

        self._init_weights()

    def _init_weights(self) -> None:
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.rul_head.weight)
        nn.init.xavier_uniform_(self.failure_head.weight)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        x : Tensor of shape (batch, seq_len, n_features)

        Returns
        -------
        dict with keys:
          "rul"          : (batch, 1)  — predicted remaining useful life
          "failure_prob" : (batch, 1)  — P(failure within window_size cycles)
        """
        # lstm_out: (batch, seq_len, hidden_size) — we only need the last step
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]  # (batch, hidden_size)

        out = self.dropout(last_hidden)
        out = self.relu(self.fc1(out))  # (batch, 64)

        rul = self.rul_head(out)  # (batch, 1)
        failure_prob = self.sigmoid(self.failure_head(out))  # (batch, 1)

        return {"rul": rul, "failure_prob": failure_prob}

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
