"""
Training pipeline for CNN-LSTM and XGBoost models.

Implements proper time-series cross-validation with an embargo period
to prevent lookahead leakage — a critical lesson from Numerai's approach.

Procedure:
  1. Load historical OHLCV from Parquet
  2. Compute features (indicators.py)
  3. Build labels: y = 1 if close_{t+15} > close_t, else 0
  4. Split: 10 months train / 1 month val (with 15-bar embargo gap)
  5. Train CNN-LSTM (PyTorch) with early stopping
  6. Train XGBoost classifier with Optuna hyperparameter search
  7. Evaluate both on validation set (accuracy, F1, Sharpe simulation)
  8. Save artifacts to models/artifacts/

Usage:
    python -m src.models.train --data data/btc_1m.parquet --output models/artifacts/
"""

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
from loguru import logger
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

from src.features.indicators import build_sequence_tensor, compute_indicators
from src.models.cnn_lstm import CNNLSTMModel
from src.models.xgboost_clf import XGBoostClassifier


LABEL_HORIZON = 15   # bars ahead (15 minutes at 1m resolution)
SEQ_LEN       = 60   # lookback window for CNN-LSTM
EMBARGO_BARS  = 15   # gap between train and val to prevent leakage


# ─────────────────────────────────────────────────────────────────────────────
#  Data preparation
# ─────────────────────────────────────────────────────────────────────────────

def load_and_label(parquet_path: str) -> pd.DataFrame:
    """Load OHLCV, compute indicators, and attach binary labels."""
    df = pd.read_parquet(parquet_path)
    df.sort_values("open_time", inplace=True)
    df.reset_index(drop=True, inplace=True)

    df = compute_indicators(df)

    # Label: 1 if price is higher in LABEL_HORIZON bars
    df["label"] = (df["close"].shift(-LABEL_HORIZON) > df["close"]).astype(int)

    # Drop rows with NaN features or labels (edges of rolling windows)
    df.dropna(subset=["macd", "rsi_14", "label"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    label_dist = df["label"].value_counts(normalize=True)
    logger.info(f"Label distribution: UP={label_dist.get(1, 0):.2%} DOWN={label_dist.get(0, 0):.2%}")
    return df


def train_val_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Time-series split: last 30 days as validation, rest as train.
    An EMBARGO_BARS gap is inserted between train and val to prevent
    label leakage through the LABEL_HORIZON future window.
    """
    n = len(df)
    val_size  = 30 * 24 * 60  # 30 days of 1-minute bars
    val_start = n - val_size
    train_end = val_start - EMBARGO_BARS  # embargo gap

    if train_end <= SEQ_LEN:
        raise ValueError("Not enough data for train/val split.")

    train_df = df.iloc[:train_end].copy()
    val_df   = df.iloc[val_start:].copy()
    logger.info(f"Train: {len(train_df):,} bars | Val: {len(val_df):,} bars")
    return train_df, val_df


# ─────────────────────────────────────────────────────────────────────────────
#  CNN-LSTM Training
# ─────────────────────────────────────────────────────────────────────────────

def build_sequences(df: pd.DataFrame, seq_len: int = SEQ_LEN):
    """Build (X_sequences, y_labels) arrays for CNN-LSTM training."""
    X, y = [], []
    for i in range(seq_len, len(df)):
        window = df.iloc[i - seq_len: i]
        seq = build_sequence_tensor(window, seq_len=seq_len)
        if seq is None:
            continue
        X.append(seq)
        y.append(int(df.iloc[i]["label"]))
    return np.array(X, dtype="float32"), np.array(y, dtype="float32")


def train_cnn_lstm(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    output_dir: str,
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: str = "cpu",
) -> CNNLSTMModel:
    logger.info("Building CNN-LSTM sequence datasets...")
    X_train, y_train = build_sequences(train_df)
    X_val,   y_val   = build_sequences(val_df)

    logger.info(f"CNN-LSTM train: {len(X_train):,} sequences | val: {len(X_val):,}")

    n_features = X_train.shape[2]
    model = CNNLSTMModel(n_features=n_features, seq_len=SEQ_LEN)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
    criterion = nn.BCELoss()

    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    val_ds   = TensorDataset(torch.tensor(X_val),   torch.tensor(y_val))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    best_val_loss = float("inf")
    patience_count = 0
    PATIENCE = 10

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb).squeeze()
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses, val_preds, val_labels = [], [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb).squeeze()
                val_losses.append(criterion(pred, yb).item())
                val_preds.extend((pred.cpu().numpy() > 0.5).astype(int))
                val_labels.extend(yb.cpu().numpy().astype(int))

        train_loss = np.mean(train_losses)
        val_loss   = np.mean(val_losses)
        val_acc    = accuracy_score(val_labels, val_preds)
        scheduler.step(val_loss)

        logger.info(
            f"Epoch {epoch:3d} | train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_count = 0
            # Save best checkpoint
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            torch.save({
                "state_dict": model.state_dict(),
                "config": {
                    "n_features": n_features,
                    "seq_len": SEQ_LEN,
                },
                "val_acc": val_acc,
                "epoch": epoch,
            }, out / "cnn_lstm_best.pt")
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                logger.info(f"Early stopping at epoch {epoch}.")
                break

    logger.success(f"CNN-LSTM best val_loss={best_val_loss:.4f}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
#  XGBoost Training with Optuna
# ─────────────────────────────────────────────────────────────────────────────

TABULAR_FEATURES = [
    "log_return", "log_return_2", "log_return_5", "rolling_vol_15",
    "macd", "macd_hist", "rsi_14", "bb_width", "ema_delta", "volume_z",
]


def prepare_tabular(df: pd.DataFrame):
    """Extract tabular feature matrix and labels from DataFrame."""
    available = [c for c in TABULAR_FEATURES if c in df.columns]
    X = df[available].fillna(0.0).values.astype("float32")
    y = df["label"].values.astype(int)
    return X, y


def train_xgboost(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    output_dir: str,
    n_trials: int = 50,
) -> XGBoostClassifier:
    X_train, y_train = prepare_tabular(train_df)
    X_val,   y_val   = prepare_tabular(val_df)

    logger.info(f"XGBoost tabular train: {len(X_train):,} | val: {len(X_val):,}")

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "max_depth":    trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":    trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma":        trial.suggest_float("gamma", 0.0, 1.0),
            "reg_alpha":    trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda":   trial.suggest_float("reg_lambda", 0.5, 2.0),
        }
        clf = XGBoostClassifier(params=params)
        clf.fit(X_train, y_train, eval_set=[(X_val, y_val)])
        preds = (clf.predict_proba_batch(X_val) > 0.5).astype(int)
        return accuracy_score(y_val, preds)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    logger.info(f"Best XGBoost val accuracy: {study.best_value:.4f}")
    best_clf = XGBoostClassifier(params=study.best_params)
    best_clf.fit(X_train, y_train, eval_set=[(X_val, y_val)])

    preds = (best_clf.predict_proba_batch(X_val) > 0.5).astype(int)
    logger.info(f"XGBoost val acc={accuracy_score(y_val, preds):.4f} "
                f"F1={f1_score(y_val, preds):.4f}")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    best_clf.save(str(out / "xgboost_best.pkl"))
    logger.success(f"XGBoost saved to {out / 'xgboost_best.pkl'}")
    return best_clf


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",   required=True, help="Path to btc_1m.parquet")
    parser.add_argument("--output", default="models/artifacts/")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--trials", type=int, default=50, help="Optuna trials for XGBoost")
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    args = parser.parse_args()

    df = load_and_label(args.data)
    train_df, val_df = train_val_split(df)

    train_cnn_lstm(train_df, val_df, args.output, epochs=args.epochs, device=args.device)
    train_xgboost(train_df, val_df, args.output, n_trials=args.trials)
