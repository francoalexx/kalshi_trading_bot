# BTC Prediction Market Trading Bot

An automated AI trading bot that predicts Bitcoin price direction and trades **Up/Down binary contracts** on [Kalshi](https://kalshi.com) using a CNN-LSTM + XGBoost ensemble model.

## How It Works

1. **Data** — Streams real-time BTC/USDT 1-minute OHLCV candles and order book depth from Binance WebSocket
2. **Features** — Computes MACD, RSI, Bollinger Band width, EMA delta, and Order Flow Imbalance (OFI) on each bar close
3. **Signal** — CNN-LSTM and XGBoost independently predict P(BTC up) over the next 15 minutes; both must agree above a confidence threshold
4. **Edge filter** — Trade only fires if `|model_probability - market_price| - fees - spread > 2%`
5. **Execution** — Places a Kalshi limit order sized by 0.5× fractional Kelly Criterion
6. **Risk** — Kill switch, 5% max position size, 10% daily loss limit, max 8 trades/hour

---

## Requirements

- Python 3.10
- A [Kalshi](https://kalshi.com) account with an API key
- A [Binance](https://binance.com) account (read-only API key, or no key — public WebSocket works without one)
- ~2 GB disk space for historical data

---

## Setup

### 1. Clone the repo

```bash
git clone <repo-url>
cd money
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment

```bash
copy .env.example .env
```

Open `.env` and fill in your values:

```env
KALSHI_API_KEY_ID=your-kalshi-key-id-here
KALSHI_PRIVATE_KEY_PATH=./kalshi_private_key.pem
KALSHI_ENV=demo
INITIAL_BANKROLL=500.0
```

### 4. Add your Kalshi private key

When you create an API key on Kalshi, they give you a `.pem` or `.txt` file containing your RSA private key. Save it to the project root:

```bash
# Rename/copy wherever Kalshi saved it:
copy "C:\path\to\your-key-file.txt" kalshi_private_key.pem
```

The file should start with `-----BEGIN PRIVATE KEY-----`.

> **Never commit this file.** It is already in `.gitignore`.

### 5. Get a Kalshi API key

1. Go to [kalshi.com](https://kalshi.com) → Settings → API
2. Create a new API key
3. Copy the **Key ID** (UUID format) into `KALSHI_API_KEY_ID` in your `.env`
4. Download the private key file and save as `kalshi_private_key.pem` (step 4 above)

Start on the **demo** environment (`KALSHI_ENV=demo`) — it uses fake money with real live markets.

---

## Phase 1 — Download Data & Train Models

### Download 12 months of BTC 1-minute data

```bash
python -m src.data.historical_downloader --start 2024-01-01 --end 2025-01-01 --output data/btc_1m.parquet
```

This downloads ~527,000 candles from Binance. Takes a few minutes.

### Train the models

```bash
python -m src.models.train --data data/btc_1m.parquet --output models/artifacts/ --epochs 50 --trials 20
```

This trains:
- **CNN-LSTM** (PyTorch) — sequence model on 60-bar OHLCV windows
- **XGBoost** — tabular classifier on technical indicators + OFI features

Artifacts are saved to `models/artifacts/`:
- `cnn_lstm_best.pt`
- `xgboost_best.pkl`

**To use GPU acceleration** (if you have an NVIDIA GPU):

```bash
# First install the CUDA-enabled PyTorch build:
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Then train with:
python -m src.models.train --data data/btc_1m.parquet --output models/artifacts/ --epochs 50 --trials 20 --device cuda
```

---

## Phase 2 — Paper Trading

Run the bot in paper mode — real live signals on real Kalshi markets, but no real money:

```bash
python -m src.main --paper
```

After each trade settles (~15 minutes), a dashboard prints to the terminal:

```
── Paper Trading Dashboard ──────────────────
  Bankroll : $512.40  (started $500.00)
  ROI      : +2.48%
  Net P&L  : $+12.40  (fees: $1.75)
  Record   : 7W / 4L  (63.6% win rate)
  Open     : 1 trades pending
  Trades   : 11 settled
─────────────────────────────────────────────
```

**Run for at least 2 weeks** and confirm:
- Win rate > 55%
- 3–8 signals per day
- No crashes

To auto-restart if it crashes (PowerShell):
```powershell
while ($true) { python -m src.main --paper; Start-Sleep -Seconds 5 }
```

Check the logs anytime:
```powershell
Get-Content logs\bot_2026-03-28.log -Tail 50
```

---

## Phase 3 — Live Trading

Once paper trading results are satisfactory, switch to live:

1. Change `KALSHI_ENV=prod` in your `.env`
2. Run without `--paper`:

```bash
python -m src.main
```

Start with a small bankroll ($500 recommended). The bot enforces:
- Max **5% of bankroll** per trade
- Max **3 open positions** simultaneously
- **10% daily loss limit** — bot shuts down automatically if hit
- **Kill switch** — press `Ctrl+C` to halt immediately (open positions settle automatically)

---

## Project Structure

```
├── src/
│   ├── data/
│   │   ├── binance_feed.py          # Binance WebSocket (OHLCV + order book)
│   │   ├── kalshi_feed.py           # Kalshi market discovery + live prices
│   │   └── historical_downloader.py # Bulk OHLCV download for training
│   ├── features/
│   │   ├── indicators.py            # MACD, RSI, Bollinger Bands, EMA
│   │   └── ofi.py                   # Order Flow Imbalance + VPIN
│   ├── models/
│   │   ├── cnn_lstm.py              # PyTorch CNN-LSTM architecture
│   │   ├── xgboost_clf.py           # XGBoost binary classifier
│   │   └── train.py                 # Training pipeline + Optuna tuning
│   ├── engine/
│   │   └── alpha.py                 # Ensemble signal + regime detector
│   ├── execution/
│   │   ├── kalshi_client.py         # RSA-PSS signed Kalshi REST v2 client
│   │   └── controller.py            # Kelly position sizer + order lifecycle
│   ├── risk/
│   │   └── manager.py               # Kill switch, daily limits, rate limiter
│   ├── paper_tracker.py             # Paper trading P&L dashboard
│   └── main.py                      # Entry point
├── tests/
│   ├── test_indicators.py
│   ├── test_ofi.py
│   └── test_kelly.py
├── .env.example                     # Environment variable template
├── .gitignore
└── requirements.txt
```

---

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```

---

## Risk Warning

This bot trades real financial instruments. Past paper trading performance does not guarantee live trading results. Only deploy capital you can afford to lose. Always start on the demo environment and validate performance before going live.
