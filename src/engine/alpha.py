"""
Alpha Engine — combines CNN-LSTM and XGBoost signals into a trading decision.

Ensemble Decision Rule:
    BUY_YES  if CNN_LSTM_prob > 0.62 AND XGBoost_prob > 0.60
    BUY_NO   if CNN_LSTM_prob < 0.38 AND XGBoost_prob < 0.40
    NO_TRADE otherwise

The 62/38 thresholds enforce a minimum confidence buffer above the fee floor.

Net edge filter:
    net_edge = |P_model - P_market| - FEE_RATE - (spread/2)
    Skip trade if net_edge < MIN_NET_EDGE (default 0.02 = 2%)

Regime detector:
    trend_strength = |EMA9 - EMA21| / ATR14
    If trend_strength < 0.5 → ranging market → reduce momentum weight,
    increase mean-reversion weight in ensemble.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional

import numpy as np
import torch
from loguru import logger

from src.models.cnn_lstm import CNNLSTMModel
from src.models.xgboost_clf import XGBoostClassifier


# ─── Kalshi fee formula at probability p ─────────────────────────────────────
# fee = 0.07 × p × (1 - p)
def kalshi_taker_fee(p: float) -> float:
    return 0.07 * p * (1 - p)


class Direction(Enum):
    BUY_YES  = "BUY_YES"
    BUY_NO   = "BUY_NO"
    NO_TRADE = "NO_TRADE"


@dataclass
class TradeSignal:
    direction:      Direction
    cnn_lstm_prob:  float   # P(up) from CNN-LSTM
    xgb_prob:       float   # P(up) from XGBoost
    ensemble_prob:  float   # weighted combination
    market_price:   float   # current YES market price from Kalshi
    net_edge:       float   # expected edge after fees and spread
    regime:         str     # "trending" or "ranging"


class AlphaEngine:
    """
    Loads trained models and produces TradeSignal objects on each bar close.

    Parameters
    ----------
    cnn_lstm_path  : path to CNN-LSTM .pt checkpoint
    xgb_path       : path to XGBoost .pkl artifact
    min_net_edge   : minimum required edge after fees/spread (default 2%)
    cnn_weight     : weight for CNN-LSTM in ensemble (XGB weight = 1 - cnn_weight)
    """

    CNN_YES_THRESHOLD = 0.62
    CNN_NO_THRESHOLD  = 0.38
    XGB_YES_THRESHOLD = 0.60
    XGB_NO_THRESHOLD  = 0.40
    TREND_STRENGTH_THRESHOLD = 0.5  # below → ranging regime

    def __init__(
        self,
        cnn_lstm_path: str,
        xgb_path: str,
        min_net_edge: float = 0.02,
        cnn_weight: float = 0.55,
        device: str = "cpu",
    ):
        self.min_net_edge = min_net_edge
        self.cnn_weight   = cnn_weight
        self.xgb_weight   = 1.0 - cnn_weight
        self.device       = device

        logger.info(f"Loading CNN-LSTM from {cnn_lstm_path}...")
        from src.models.cnn_lstm import load_cnn_lstm
        self.cnn_lstm: CNNLSTMModel = load_cnn_lstm(cnn_lstm_path, device=device)

        logger.info(f"Loading XGBoost from {xgb_path}...")
        self.xgboost: XGBoostClassifier = XGBoostClassifier.load(xgb_path)

        logger.success("Alpha Engine ready.")

    def evaluate(
        self,
        sequence: np.ndarray,
        features: Dict[str, float],
        market_price: float,
        spread: float,
    ) -> TradeSignal:
        """
        Generate a trading signal.

        Parameters
        ----------
        sequence     : np.ndarray, shape (60, n_features) — for CNN-LSTM
        features     : dict of tabular features — for XGBoost
        market_price : current Kalshi YES ask price (0-1 scale)
        spread       : current Kalshi YES bid-ask spread (0-1 scale)
        """
        # ── Regime detection ─────────────────────────────────────────────
        regime = self._detect_regime(features)

        # ── CNN-LSTM inference ───────────────────────────────────────────
        x_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(self.device)
        cnn_prob = self.cnn_lstm.predict_proba(x_tensor)

        # ── XGBoost inference ────────────────────────────────────────────
        xgb_prob = self.xgboost.predict_proba(features)

        # ── Regime-adjusted ensemble weights ────────────────────────────
        if regime == "ranging":
            # In ranging market: down-weight momentum model (CNN-LSTM)
            # and up-weight the mean-reversion signals captured by XGBoost RSI/BB
            cnn_w = self.cnn_weight * 0.7
            xgb_w = 1.0 - cnn_w
        else:
            cnn_w = self.cnn_weight
            xgb_w = self.xgb_weight

        ensemble_prob = cnn_w * cnn_prob + xgb_w * xgb_prob

        # ── Direction logic ──────────────────────────────────────────────
        direction = Direction.NO_TRADE
        if cnn_prob > self.CNN_YES_THRESHOLD and xgb_prob > self.XGB_YES_THRESHOLD:
            direction = Direction.BUY_YES
        elif cnn_prob < self.CNN_NO_THRESHOLD and xgb_prob < self.XGB_NO_THRESHOLD:
            direction = Direction.BUY_NO

        # ── Net edge calculation ─────────────────────────────────────────
        # For YES bet: edge = P_model - P_market (if positive)
        # For NO  bet: edge = (1 - P_model) - (1 - P_market) = P_market - P_model
        if direction == Direction.BUY_YES:
            gross_edge = ensemble_prob - market_price
        elif direction == Direction.BUY_NO:
            gross_edge = market_price - ensemble_prob
        else:
            gross_edge = 0.0

        fee = kalshi_taker_fee(market_price)
        net_edge = gross_edge - fee - (spread / 2)

        if direction != Direction.NO_TRADE and net_edge < self.min_net_edge:
            logger.debug(
                f"Signal filtered: direction={direction.value} "
                f"gross_edge={gross_edge:.4f} net_edge={net_edge:.4f} < {self.min_net_edge}"
            )
            direction = Direction.NO_TRADE

        signal = TradeSignal(
            direction=direction,
            cnn_lstm_prob=cnn_prob,
            xgb_prob=xgb_prob,
            ensemble_prob=ensemble_prob,
            market_price=market_price,
            net_edge=net_edge,
            regime=regime,
        )

        if direction != Direction.NO_TRADE:
            logger.info(
                f"SIGNAL: {direction.value} | cnn={cnn_prob:.3f} xgb={xgb_prob:.3f} "
                f"ensemble={ensemble_prob:.3f} net_edge={net_edge:.4f} regime={regime}"
            )

        return signal

    def _detect_regime(self, features: Dict[str, float]) -> str:
        """
        Classify market regime as 'trending' or 'ranging'.

        Uses EMA delta (proxy for EMA9-EMA21 distance) and Bollinger Band
        width as regime indicators.

        trend_strength = |ema_delta| (already normalised by price in indicators.py)
        If |ema_delta| < TREND_STRENGTH_THRESHOLD threshold → ranging
        """
        ema_delta_abs = abs(features.get("ema_delta", 0.0))
        bb_width      = features.get("bb_width", 0.05)

        # Compound score: low directional delta + narrow BB → ranging
        trend_score = ema_delta_abs * 100 + bb_width * 10
        regime = "trending" if trend_score > self.TREND_STRENGTH_THRESHOLD else "ranging"
        return regime
