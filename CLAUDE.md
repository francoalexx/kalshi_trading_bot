# BTC Prediction Market Trading Bot — Project Memory

> **Maintenance rule:** Update this file at the end of every coding session.
> Any change to a threshold, guard, bug fix, architecture decision, or lesson
> learned from live/paper trading must be recorded here before closing the
> session. Future Claude instances load this file cold — if it isn't here,
> it will be forgotten.

---

## Project Vision
Automated, self-improving bot that trades BTC **Up/Down binary contracts** on
**Kalshi KXBTC15M** (15-minute markets) by predicting short-term BTC price
direction using a dual ML ensemble. Goal: compound a $500 bankroll to $100+/day
revenue through continuous learning, disciplined risk management, and statistical
promotion of improved models.

**Current environment:** `KALSHI_ENV=prod` (live trading on Kalshi)
**Starting bankroll:** $100 (live test) | **Target:** $100/day at ~$3,000–$5,000 bankroll

---

## Architecture

```
Binance WS (1m OHLCV + order book depth)
        │
        ▼
Feature Engine (src/features/)
  indicators.py  → MACD(12,26,9), RSI(14), BB-width, EMA9−EMA21, log-returns, vol
  ofi.py         → OFI, VPIN, aggressor_ratio
        │
        ▼
AlphaEngine (src/engine/alpha.py)
  OTM guard: skip if market_price < 0.15 or > 0.85  ← CRITICAL, see Lessons
  CNN-LSTM (60-bar sequence)  → P(up)
  XGBoost  (13 tabular feats) → P(up)
  Ensemble → p_yes (adjusted for strike_type: greater/less)
  Regime detector (trend_score) → trending | ranging (adjusts CNN weight)
  HeuristicsEngine.apply()     → rule-based signal filter/modifier
        │
        ▼
Edge filter: net_edge = |P_model − P_market| − fee(0.07×p×(1−p)) − spread/2
        │  skip if net_edge < MIN_NET_EDGE
        ▼
RiskManager (src/risk/manager.py)
  Kill switch · max_open_positions · daily_loss_limit · max_trades_per_hour
        │
        ▼
ExecutionController (src/execution/controller.py)
  0.5× fractional Kelly sizing → limit order via KalshiClient
  Poll fill every 5s → cancel at T−90s before expiry
        │
        ▼
TradeJournal (src/learning/trade_journal.py) — SQLite: bars + trades + heuristics_log
PostmortemEngine → failure classification every 50 settled trades
AttributionAnalyzer → SHAP feature alpha ranking
ShadowModelManager → parallel XGBoost with Optuna candidate params
PromotionOrchestrator → Mann-Whitney gate (p<0.05) → auto-promote or revert
```

---

## Key Thresholds & Parameters

| Parameter | Value | Location |
|---|---|---|
| Bankroll | $500 | `INITIAL_BANKROLL` (.env) |
| Max position | 5% bankroll | `MAX_POSITION_PCT` (.env) |
| Daily loss limit | 10% | `DAILY_LOSS_LIMIT_PCT` (.env) |
| Max open positions | 5 | `MAX_OPEN_POSITIONS` (.env) |
| Max trades/hour | 20 | `MAX_TRADES_PER_HOUR` (.env) |
| Min net edge | 1% | `MIN_NET_EDGE` (.env) |
| **XGB YES threshold** | **0.54** | hardcoded in alpha.py |
| **XGB NO threshold** | **0.46** | hardcoded in alpha.py |
| CNN YES/NO threshold | 0.50 | hardcoded in alpha.py |
| **OTM guard LOW** | **0.15** | hardcoded in alpha.py |
| **OTM guard HIGH** | **0.85** | hardcoded in alpha.py |
| **Feed ATM reject LOW** | **0.15** | hardcoded in kalshi_feed.py |
| **Feed ATM reject HIGH** | **0.85** | hardcoded in kalshi_feed.py |
| Cancel deadline | T−90s | hardcoded in controller.py |
| Analysis interval | 50 trades | `ANALYSIS_INTERVAL` (.env) |
| Shadow min trades | 50 | `SHADOW_MIN_TRADES` (.env) |

---

## Platform: Kalshi
- **Series:** KXBTC15M (15-minute BTC Up/Down binary contracts)
- **API base:** `https://api.elections.kalshi.com/trade-api/v2`
- **Auth:** RSA-PSS, `salt_length=DIGEST_LENGTH` (32 bytes), SHA-256
- **Fee formula:** `0.07 × p × (1 − p)` per contract at price p
- **Breakeven:** >53.5% win rate at 50¢ contracts
- **Market selection:** Prefer liquid (YES bid > 0), then closest to ATM (mid ≈ 0.50)
- **ATM validity window:** `kalshi_feed.get_active_market()` returns `None` if mid is
  outside [0.15, 0.85] — bot waits for next 15-min market rather than trading OTM
- **strike_type:** `greater_or_equal` → YES wins if BTC up; `less` → YES wins if BTC down
  - In alpha.py: `p_yes = 1 − ensemble_prob` for bearish markets
- **Settlement parsing:** Kalshi sometimes returns result as `'[9.56E-2]'` (bracketed
  string) — settlement monitor in main.py strips brackets before comparing "yes"/"no"

---

## ML Models

### CNN-LSTM (`src/models/cnn_lstm.py`)
- Input: `(1, 60, ~20_features)` tensor
- Architecture: Conv1D(32,3) → MaxPool → LSTM(64) → LSTM(32) → Dense(1, sigmoid)
- Artifact: `models/artifacts/cnn_lstm_best.pt`
- Role: momentum/sequence pattern detection

### XGBoost (`src/models/xgboost_clf.py`)
- Input: 13 tabular features (see `FEATURE_NAMES` in xgboost_clf.py)
- Artifact: `models/artifacts/xgboost_best.pkl`
- Role: mean-reversion + regime-specific signals
- Has `retrain(X, y, params)` method for shadow model retraining

### Training
```bash
python -m src.data.historical_downloader --start 2024-01-01 --end 2025-01-01 \
  --output data/btc_1m.parquet
python -m src.models.train --data data/btc_1m.parquet --output models/artifacts/ \
  --epochs 50 --trials 20
```
- Uses time-series CV with 10-bar embargo (no lookahead)
- Optuna tunes XGBoost params; CNN uses fixed architecture + early stopping

---

## Continuous Learning System (`src/learning/`)

### Trade Journal (SQLite: `data/trade_journal.db`)
- **bars:** every 1m bar — all 13 features + signal outputs + market context
- **trades:** every filled order — execution quality + outcome after settlement
- **heuristics_log:** audit trail of rule changes and model promotions

### Postmortem & Attribution (every 50 settled trades)
- `postmortem.py`: classifies losses → `ALPHA_DECAY` | `FALSE_POSITIVE` | `MARKET_NOISE`
- `attribution.py`: SHAP TreeExplainer → ranks features by mean |SHAP|; detects
  degrading alpha (>50% drop vs 3-cycle mean)

### Heuristics Registry (`data/heuristics.json`)
- Rules applied inside `alpha.evaluate()` via `HeuristicsEngine.apply()`
- Rule types: `skip_trade` | `scale_position` | `reduce_edge_requirement`
- Auto-disabled when confidence < 0.55; promoted to "stable" at ≥ 0.65 + 50 trades

### Shadow Model & Promotion (`shadow_model.py`, `adaptation.py`)
- Weekly Optuna study (background thread) → candidate XGBoost params
- `ShadowModelManager` watches every bar (no real/paper orders)
- `PromotionOrchestrator`: promote only if Mann-Whitney p < 0.05 AND better Sharpe
  AND drawdown ≤ live × 1.2
- 20-trade probation post-promotion; auto-revert if Sharpe drops >30%
- Model checkpoints: `xgboost_best.pkl`, `xgboost_prev.pkl`, `xgboost_prev2.pkl`

---

## Development Commands

```bash
# Paper trade on real Kalshi KXBTC15M markets (no real money)
python -m src.main --paper

# Live trading
python -m src.main

# Custom model paths
python -m src.main --cnn models/artifacts/cnn_lstm_best.pt \
                   --xgb models/artifacts/xgboost_best.pkl --paper

# Run all unit tests
pytest tests/ -v

# Diagnose Kalshi API connectivity + market availability
python diagnose.py

# Export trade history CSV from SQLite (rerunnable)
python export_trades.py

# Run deep-dive trade analysis → data/analysis_report.md
python analysis.py

# Query the trade journal
python -c "
from src.learning.trade_journal import TradeJournal
j = TradeJournal()
print(j.get_recent_trades(20).to_string())
"

# Check heuristics rules
cat data/heuristics.json

# View SHAP history
cat data/shap_history.json

# Trigger Optuna study manually
python -c "
from src.learning.adaptation import run_optuna_study
from src.learning.trade_journal import TradeJournal
print(run_optuna_study(TradeJournal(), n_trials=20))
"
```

---

## Risk Controls (Never Bypass)
1. **Kill switch** (`src/risk/manager.py`) — fires on: daily loss limit breach, SIGINT,
   uncaught exception. Requires manual `reset_kill_switch()` to resume.
2. **T−90s cancel** — all unfilled orders cancelled 90 seconds before market expiry.
3. **No market orders** — limit orders only (Kalshi).
4. **OTM guard** — never trade when `market_price < 0.15` or `market_price > 0.85`.
   The ML model signal has no bearing on OTM contract outcomes (see Lessons).
5. **Paper before live** — always validate a new model/heuristic in paper mode for ≥2 days.
6. **Model promotion gate** — never overwrite `xgboost_best.pkl` without passing
   the Mann-Whitney promotion test in `adaptation.py`.

---

## Lessons Learned (Critical — Read Before Changing Thresholds)

### OTM Contract Trap (discovered 2026-03-31)
**What happened:** 136-trade paper run produced 9.6% win rate and −490% ROI.
Bot was buying YES/NO contracts at 1¢–3¢ (deeply OTM).

**Root cause:** The ML model outputs `P(BTC direction in 15 min)` — not
`P(BTC finishes above/below a specific strike)`. For ATM markets (~50¢)
these are roughly equivalent. For a 1¢ contract they are completely different
questions. A 1¢ NO contract in a "greater" market means the strike is already
far above current BTC price; the model's directional signal is irrelevant.

**Compounding factor:** Kelly formula sees a false edge of ~0.54 on OTM markets
(model says 55%, market prices 1%) and sizes to the maximum cap every time
(2,500–3,125 contracts × $0.01 = $25/trade). Every trade loses $25.

**Fix applied:**
- `alpha.py`: Hard block if `market_price < 0.15` or `market_price > 0.85`
- `kalshi_feed.py`: `get_active_market()` returns `None` when mid drifts outside
  [0.15, 0.85], forcing the bot to wait for the next market cycle
- **Never remove these guards** without a principled alternative that handles
  the P(direction) ≠ P(strike crossed) problem for OTM markets.

### XGB Threshold Sensitivity (discovered 2026-03-31)
Lowering XGB thresholds from 0.54/0.46 → 0.52/0.48 increased signal frequency
but generated too many marginal trades below the fee floor. Reverted to 0.54/0.46.
Do not lower thresholds again until OTM-clean paper results confirm model edge.

### Kalshi Settlement Parsing (discovered 2026-03-31)
Kalshi API occasionally returns the `result` field as `'[9.558824E-2]'` — a
float wrapped in a JSON array rendered as a string. Settlement monitor now
strips brackets before checking for `"yes"`/`"no"`. If settlements stop working,
check this parsing first.

### signal_edge / kelly_fraction not saved to journal (fixed 2026-04-04)
`OrderResult` dataclass was missing `signal_edge` and `kelly_fraction` fields.
`log_trade()` always saved `None`, causing the postmortem engine to classify
every loss as `ALPHA_DECAY` (since `None` coerces to 0.0 < threshold).
Fix: added both fields to `OrderResult`, populated from `signal` and `_kelly_size()`
in `controller.py`. `_kelly_size()` now returns `(position_usd, f_half)` tuple.
Failure classifications will be meaningful from trade #119 onward.

### Console log level raised to INFO (2026-04-04)
`src/main.py` stderr logger changed from `DEBUG` to `INFO`. Debug-level candle
and market-poll messages now go to the log file only. Console shows trades,
signals, warnings, and errors only.

### Live settlement monitor was missing (fixed 2026-04-04)
**What happened:** Pre-live audit found that `monitor_paper_settlements()` only ran
in paper mode. In live mode, after an order filled, the bot completely forgot about
it — no P&L tracking, no bankroll update, no risk manager integration. This meant:
- `risk.record_settlement()` was never called (daily loss limit inert)
- Controller bankroll never updated (Kelly sizing became stale)
- No journal logging or postmortem analysis for live trades
- No email notifications for live settlements

**Fix:** Replaced `monitor_paper_settlements()` with unified `monitor_settlements()`
that works for both paper and live trades. New `PendingTrade` dataclass tracks all
open trades. `_settle_trade()` handles the full settlement flow: P&L calculation,
risk manager update, controller bankroll sync, journal + postmortem logging, and
email notifications.

### KalshiClient env parameter bug (fixed 2026-04-04)
`KalshiClient.__init__` had `env: str = "demo"` as default, which shadowed the
`.env` file's `KALSHI_ENV` setting. The `env or os.getenv(...)` fallback never
triggered because `"demo"` is truthy. Fixed by changing default to `None`.
Also fixed `diagnose.py` to pass `env` explicitly.

### Live mode credential validation (added 2026-04-04)
Bot now exits with a clear error if started in live mode without a valid Kalshi
private key, instead of silently entering DRY_RUN mode where orders are faked.

---

## File Map (Critical Paths)

| Component | File |
|---|---|
| Entry point | `src/main.py` |
| Alpha engine + OTM guard | `src/engine/alpha.py` |
| Kalshi client | `src/execution/kalshi_client.py` |
| Execution + Kelly | `src/execution/controller.py` |
| Risk limits | `src/risk/manager.py` |
| Market feed + ATM filter | `src/data/kalshi_feed.py` |
| Price feed | `src/data/binance_feed.py` |
| Indicators | `src/features/indicators.py` |
| OFI | `src/features/ofi.py` |
| CNN-LSTM | `src/models/cnn_lstm.py` |
| XGBoost | `src/models/xgboost_clf.py` |
| Training pipeline | `src/models/train.py` |
| Trade journal | `src/learning/trade_journal.py` |
| Heuristics engine | `src/learning/heuristics.py` |
| SHAP attribution | `src/learning/attribution.py` |
| Failure classifier | `src/learning/postmortem.py` |
| Shadow model | `src/learning/shadow_model.py` |
| Optuna + promotion | `src/learning/adaptation.py` |
| Paper P&L tracker | `src/paper_tracker.py` |
| Email notifications | `src/notifications.py` |
| Trade history export | `export_trades.py` |
| Trade analysis | `analysis.py` |
| Diagnostics | `diagnose.py` |
| Config | `.env` (gitignored) |
| Historical data | `data/btc_1m.parquet` |
| Trade history DB | `data/trade_journal.db` |
| Trade history CSV | `data/trade_history.csv` |
| Analysis report | `data/analysis_report.md` |
| Heuristics rules | `data/heuristics.json` |

---

## Progress & Roadmap

### ✅ Complete
- [x] Kalshi RSA-PSS authentication (PROD URL + DIGEST_LENGTH salt)
- [x] KXBTC15M market discovery and ATM selection
- [x] CNN-LSTM + XGBoost dual ensemble with regime detection
- [x] 0.5× fractional Kelly position sizing
- [x] Order lifecycle (place → poll → fill → T−90s cancel)
- [x] Paper trading on real Kalshi markets (not simulated)
- [x] Real Kalshi settlement polling (result field after expiry)
- [x] Risk manager (kill switch, daily limits, rate limits)
- [x] Continuous learning system (journal, SHAP, postmortem, shadow, Optuna)
- [x] Heuristics rules engine with confidence decay
- [x] Bearish market (strike_type=less) signal direction inversion
- [x] OTM guard (market_price bounds [0.15, 0.85]) — critical bug fix
- [x] ATM staleness filter in KalshiFeed (withhold OTM markets)
- [x] Kalshi settlement result parsing fix (bracketed float strings)
- [x] signal_edge and kelly_fraction now correctly saved to trade journal
- [x] Console logging raised to INFO (debug noise to file only)
- [x] Trade history CSV export (`export_trades.py` with timestamps)
- [x] Deep-dive trade analysis report (`analysis.py` → `data/analysis_report.md`)
- [x] Email notifications on trade settlement via Resend (`src/notifications.py`)
- [x] Unified settlement monitor for both paper AND live modes
- [x] Live mode: risk manager, journal, postmortem, email all wired up
- [x] Live mode credential validation (exits if no private key)
- [x] KalshiClient env parameter bug fix (KALSHI_ENV now respected)
- [x] Paper run complete — 124 trades (56.5% win rate, +$2,082 net from $500)

### 🔄 In Progress
- [ ] First live trading run — $100 bankroll on prod (started 2026-04-04)
- [ ] Monitor first 5-10 live trades for correct settlement flow
- [ ] Accumulate live trades for comparison with paper performance

### 🗓 Next Milestones
- [ ] **Phase 4:** First SHAP report — identify top 3 alpha features
- [ ] **Phase 5:** First Optuna study after 100 settled trades
- [ ] **Phase 6:** Shadow model promotion if candidate params improve Sharpe
- [ ] **Phase 7:** Scale live bankroll to $500 after stable live results
- [ ] **Phase 8:** Compound bankroll toward $1,500 target for $30–$40/day EV

### 💰 Revenue Milestones
| Bankroll | Max position | EV/trade @55% win | Trades/day | Est. daily |
|---|---|---|---|---|
| $500 (now) | $25 | ~$1.50 | 5–15 ATM only | $7–22 |
| $1,500 | $75 | ~$4.50 | 20 | ~$90 |
| $3,000 | $150 | ~$9.00 | 15 | ~$135 |
| $5,000 | $250 | ~$15.00 | 10 | ~$150 |
