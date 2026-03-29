"""
CNN-LSTM model for binary BTC price direction prediction.

Architecture:
    Conv1D(32, kernel=3, ReLU) → MaxPool(2) → Dropout(0.2)
    → LSTM(64, return_sequences=True) → LSTM(32)
    → Dense(16, ReLU) → Dense(1, Sigmoid)

Input shape:  (batch, seq_len=60, n_features=13)
Output:       P(BTC_up) ∈ [0, 1] over the next 15 minutes

Based on architecture achieving 82.44% accuracy on binary BTC direction
prediction (CNN-LSTM with Boruta feature selection, 2024 MDPI study).
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional


class CNNLSTMModel(nn.Module):
    """
    Convolutional-LSTM hybrid for time-series binary classification.

    The CNN front-end learns local patterns (e.g., candle formations, MACD
    crossovers) while the stacked LSTM captures longer sequential dependencies.
    """

    def __init__(
        self,
        n_features: int = 13,
        seq_len: int = 60,
        cnn_filters: int = 32,
        cnn_kernel: int = 3,
        lstm1_hidden: int = 64,
        lstm2_hidden: int = 32,
        dense_hidden: int = 16,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.n_features = n_features

        # ── CNN block ────────────────────────────────────────────────────
        # Input: (batch, seq_len, n_features) → permute to (batch, n_features, seq_len)
        self.conv1 = nn.Conv1d(
            in_channels=n_features,
            out_channels=cnn_filters,
            kernel_size=cnn_kernel,
            padding=cnn_kernel // 2,  # same padding
        )
        self.relu_conv = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.dropout_cnn = nn.Dropout(dropout)

        # After MaxPool(2): seq dimension halved → seq_len // 2
        cnn_out_len = seq_len // 2

        # ── LSTM block ───────────────────────────────────────────────────
        # Input to LSTM: (batch, cnn_out_len, cnn_filters)
        self.lstm1 = nn.LSTM(
            input_size=cnn_filters,
            hidden_size=lstm1_hidden,
            batch_first=True,
            dropout=0.0,  # dropout only valid with num_layers>1; applied via Dropout layer instead
        )
        self.lstm2 = nn.LSTM(
            input_size=lstm1_hidden,
            hidden_size=lstm2_hidden,
            batch_first=True,
        )
        self.dropout_lstm = nn.Dropout(dropout)

        # ── Dense head ───────────────────────────────────────────────────
        self.fc1 = nn.Linear(lstm2_hidden, dense_hidden)
        self.relu_fc = nn.ReLU()
        self.fc2 = nn.Linear(dense_hidden, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor, shape (batch, seq_len, n_features)

        Returns
        -------
        torch.Tensor, shape (batch, 1) — P(BTC_up)
        """
        # CNN expects (batch, channels, length)
        x = x.permute(0, 2, 1)               # → (batch, n_features, seq_len)
        x = self.relu_conv(self.conv1(x))     # → (batch, cnn_filters, seq_len)
        x = self.pool(x)                      # → (batch, cnn_filters, seq_len//2)
        x = self.dropout_cnn(x)

        # LSTM expects (batch, seq, input_size)
        x = x.permute(0, 2, 1)               # → (batch, seq_len//2, cnn_filters)
        x, _ = self.lstm1(x)                  # → (batch, seq_len//2, lstm1_hidden)
        x, _ = self.lstm2(x)                  # → (batch, seq_len//2, lstm2_hidden)
        x = self.dropout_lstm(x[:, -1, :])    # take last timestep

        x = self.relu_fc(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

    def predict_proba(self, x: torch.Tensor) -> float:
        """
        Convenience method: run inference and return scalar P(BTC_up).

        Parameters
        ----------
        x : torch.Tensor, shape (1, seq_len, n_features) or (seq_len, n_features)
        """
        self.eval()
        with torch.no_grad():
            if x.dim() == 2:
                x = x.unsqueeze(0)
            return float(self.forward(x).squeeze())


def load_cnn_lstm(model_path: str, device: str = "cpu") -> CNNLSTMModel:
    """Load a saved CNN-LSTM model from a .pt checkpoint."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=device)
    config = checkpoint.get("config", {})
    model = CNNLSTMModel(**config)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model
