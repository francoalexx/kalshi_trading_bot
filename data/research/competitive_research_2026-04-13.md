# Kalshi BTC 15-Minute Trading Bot — Competitive Research & Strategy Plan

## Context

We're running a BTC prediction market bot on Kalshi's KXBTC15M (15-minute binary contracts). The bot uses a CNN-LSTM + XGBoost ensemble to predict BTC direction, then trades YES/NO contracts. Current challenges: the retrained XGBoost produces overconfident signals creating false "huge edges" on near-OTM markets, and we want to understand what other builders are doing to find new alpha sources or validate our approach. This research surveyed 30+ GitHub repos, academic papers, blog posts, and community discussions.

---

## A. Strategy Catalog

### 1. Momentum / Trend Continuation
- **Source:** [DeweyMarco/simple-kalshi-bot](https://github.com/DeweyMarco/simple-kalshi-bot)
- **Core idea:** Buy the same side that won the previous 15-min market (autocorrelation bet), or follow BTC spot momentum over the last 60 seconds.
- **Signals:** Previous settlement result, 60-second BTC price change (Coinbase API)
- **Claimed edge:** Trend persistence across consecutive 15-min windows
- **Complexity:** Low
- **Relevance to us:** Medium — we already use momentum signals (MACD, EMA crossover) but don't explicitly track previous-market autocorrelation. Worth testing as a heuristic rule.
- **Caveat:** No published backtest results. Contradicts random walk theory at 15-min scale.

### 2. Multi-LLM Consensus Trading
- **Source:** [ryanfrigo/kalshi-ai-trading-bot](https://github.com/ryanfrigo/kalshi-ai-trading-bot) (~79 stars)
- **Core idea:** 5 frontier LLMs (Claude, Gemini, GPT, DeepSeek, Grok) independently analyze each market. Only trade when consensus confidence > 45%.
- **Signals:** LLM-generated probability estimates, "Safe Compounder" mode targeting NO-side trades
- **Claimed edge:** 74% win rate on NO-side "Safe Compounder" trades
- **Complexity:** Medium (API costs ~$10/day)
- **Relevance to us:** Low-Medium — interesting for longer-duration event markets, but LLMs are too slow for 15-min crypto windows where the signal is price action, not fundamentals.
- **Caveat:** No verified live P&L published. LLM latency (seconds) vs market speed (milliseconds) is a fundamental mismatch for 15-min contracts.

### 3. Deep Fundamental Research + Edge Detection
- **Source:** [OctagonAI/kalshi-deep-trading-bot](https://github.com/OctagonAI/kalshi-deep-trading-bot) (~78 stars)
- **Core idea:** AI generates independent probability estimates with ranked "price drivers," then compares vs live order book to compute edge. Five-gate risk engine filters trades.
- **Signals:** AI-generated probabilities, catalyst calendars, confidence metrics
- **Claimed edge:** Information advantage via deep research
- **Complexity:** High
- **Relevance to us:** Low — designed for longer-duration event markets (politics, economics), not 15-min crypto.

### 4. Cross-Platform Arbitrage (Polymarket <> Kalshi)
- **Source:** [CarlosIbCu/polymarket-kalshi-btc-arbitrage-bot](https://github.com/CarlosIbCu/polymarket-kalshi-btc-arbitrage-bot) (~40 stars)
- **Core idea:** When YES on Kalshi + NO on Polymarket < $1.00, buy both sides for guaranteed profit.
- **Signals:** Real-time order book monitoring on both platforms
- **Claimed edge:** 12-20% monthly (pre-fees)
- **Complexity:** High (dual-platform infrastructure, no orphan legs)
- **Relevance to us:** Low — spreads have compressed to 1-2% in 2026 as institutional capital entered. Requires microsecond execution. Not viable at retail scale.

### 5. Latency Arbitrage (Binance Lead -> Kalshi Lag)
- **Source:** [learningworship/polymarket-latency-bot](https://github.com/learningworship/polymarket-latency-bot)
- **Core idea:** Exploit 2-10 second lag between Binance BTC price moves and prediction market repricing. Enter when Binance price shift > 0.4%.
- **Signals:** Binance real-time price vs 30-second historical baseline
- **Claimed edge:** Speed advantage in repricing lag
- **Complexity:** Medium
- **Relevance to us:** Medium-High — we already have Binance WebSocket data. The question is whether Kalshi reprices slower than Binance moves on 15-min contracts. Worth investigating with our price logger data.
- **Caveat:** Edge may be minimal if Kalshi market makers are fast.

### 6. Market Making (Avellaneda-Stoikov)
- **Source:** [rodlaf/KalshiMarketMaker](https://github.com/rodlaf/KalshiMarketMaker), [nikhilnd/kalshi-market-making](https://github.com/nikhilnd/kalshi-market-making)
- **Core idea:** Post two-sided quotes around estimated fair probability. Adjust spread asymmetrically based on inventory. Earn the spread at volume.
- **Signals:** Fair probability estimate, inventory position, Cauchy distribution for fat-tailed pricing
- **Claimed edge:** nikhilnd achieved 20.3% return ($6.80 on $33.40) over 51 trades
- **Complexity:** High (requires significant capital, multiple simultaneous markets)
- **Relevance to us:** Low — requires $10k+ capital and 20+ simultaneous positions. Our $100-500 bankroll is too small.

### 7. VWAP-Momentum + Orderbook Skew
- **Source:** [Poly-Tutor bots](https://github.com/Poly-Tutor/polymarket-5min-15min-1hour-arbitrage-trading-bot-tools)
- **Core idea:** Three strategies: (1) VWAP deviation + momentum on 15-min windows, (2) final-minute orderbook skew confirmation, (3) live BTC vs price-to-beat comparison.
- **Signals:** VWAP, momentum, orderbook depth imbalance in final minute before settlement
- **Claimed edge:** Momentum + orderbook confluence
- **Complexity:** Medium
- **Relevance to us:** High — orderbook skew in the final minute before Kalshi settlement is a signal we're not using. If Kalshi's order book tilts heavily YES or NO in the last 1-2 minutes, that's informed trader flow.

### 8. NLP Sentiment + Statistical Arbitrage + GARCH Volatility
- **Source:** [yllvar/Kalshi-Quant-TeleBot](https://github.com/yllvar/Kalshi-Quant-TeleBot) (~50 stars)
- **Core idea:** Three concurrent strategies: (1) NLP sentiment scoring, (2) cointegration-based stat arb, (3) GARCH volatility modeling for entry/exit timing.
- **Signals:** Social media sentiment, cointegrated event pairs, volatility regime
- **Claimed edge:** Multi-strategy diversification
- **Complexity:** High
- **Relevance to us:** Medium — GARCH volatility modeling is interesting. Our regime detector is simple (trend_score threshold); GARCH would give us a more principled volatility estimate that could improve position sizing and signal filtering.

### 9. Jump Mean-Reversion (JMR)
- **Source:** Academic research, [DEV Community writeup](https://dev.to/ayratmurtazin/intraday-volatility-jump-mean-reversion-trading-strategy-for-btc-usd-44lf)
- **Core idea:** After BTC makes an extreme intraday move (a "jump"), systematically trade against it on the assumption it will partially revert.
- **Signals:** Intraday volatility spike detection, z-score of 15-min returns
- **Claimed edge:** Overshooting after volatility spikes creates predictable reversals
- **Complexity:** Low-Medium
- **Relevance to us:** High — our XGBoost already uses RSI and Bollinger Bands (mean-reversion signals), but we don't explicitly detect and trade jumps/spikes. A jump detector could be a high-value heuristic rule.

### 10. Liquidation Cascade Detection
- **Source:** CoinGlass, CoinAnk, academic research
- **Core idea:** Monitor Binance/Deribit liquidation feeds. When cascading liquidations exceed a threshold (e.g., >$50M in 15 min), predict a reversal.
- **Signals:** Real-time liquidation data from CoinGlass API or Binance WebSocket
- **Claimed edge:** Liquidation cascades create forced selling/buying that overshoots fair value
- **Complexity:** Medium (need additional data feed)
- **Relevance to us:** High — this is a genuinely different signal source from what we have. Liquidation cascades are mechanical (forced position closing), not discretionary, so they create predictable short-term price behavior.

---

## B. Key Academic Findings

### The Hard Truth About Kalshi
- **Average pre-fee return across 300,000+ contracts: -20%** (Karl Whelan, UCD)
- Makers lose ~10% on average; takers lose ~32%
- Strong favorite-longshot bias: contracts <$0.10 lose >60% of capital
- Most profitable approach: selling overpriced longshots (high-probability NO trades)
- Short-term ML prediction accuracy on BTC: 50.9%-56% (barely above random)
- Profitable prediction requires >53.5% win rate after Kalshi's fee formula

### What This Means For Us
Our retrained XGBoost hitting 59.7% WR at (0.55/0.45) thresholds on validation data is genuinely above the academic ceiling — IF it holds in live trading. The MAX_NET_EDGE filter is critical because the academic data confirms that large apparent edges are almost always the favorite-longshot bias in disguise.

---

## C. Ideas Worth Exploring (Ranked by Expected Value)

### 1. Liquidation Cascade Signal (NEW DATA SOURCE)
- **What:** Add Binance liquidation stream as a feature. When large cascading liquidations occur, the model gets a strong mean-reversion signal.
- **Why it might work:** Liquidations are mechanical forced trades, not discretionary. They create predictable overshooting. This is a genuinely different signal from price/volume technicals.
- **Data needed:** Binance WebSocket `forceOrder` stream (free, real-time)
- **Effort:** Medium (1-2 sessions) — new data feed + feature engineering + retrain
- **Risks:** Liquidation cascades may be too rare for 15-min windows. Need to validate frequency.

### 2. Latency Arbitrage Detection
- **What:** Measure the lag between Binance BTC spot moves and Kalshi contract repricing. If Kalshi lags by 2-10 seconds on large moves, enter during the lag window.
- **Why it might work:** We already have both Binance and Kalshi price data logging. If there's a consistent lag, it's mechanical alpha.
- **Data needed:** Already collecting (contract_prices.db + Binance WebSocket)
- **Effort:** Low (1 session) — analysis of existing data, then a simple heuristic rule
- **Risks:** Kalshi market makers may be fast enough that no exploitable lag exists. Need to measure first.

### 3. Jump/Spike Mean-Reversion Heuristic
- **What:** Detect when BTC makes an extreme move in the last 5-10 minutes (z-score > 2 on 15-min returns). Bias toward the reversal direction.
- **Why it might work:** Academically documented that extreme intraday moves overshoot. Our XGBoost already captures some of this via RSI/BB, but an explicit jump detector as a heuristic rule could boost it.
- **Data needed:** Already have (Binance 1-min candles)
- **Effort:** Low (1 session) — new heuristic rule in heuristics.json
- **Risks:** Not all jumps revert in 15 minutes. Need to validate on historical data.

### 4. Previous-Market Autocorrelation
- **What:** Track whether the previous KXBTC15M market settled YES or NO. Test if there's autocorrelation (momentum or mean-reversion across consecutive 15-min windows).
- **Why it might work:** Simple Kalshi Bot uses this. If BTC is trending, consecutive markets may settle the same way.
- **Data needed:** Already have (settlement history in trade journal)
- **Effort:** Very low (analysis + optional heuristic rule)
- **Risks:** May be noise. Easy to test though.

### 5. Orderbook Skew in Final Minutes
- **What:** Monitor Kalshi order book depth imbalance in the last 1-3 minutes before settlement. Heavy YES bids vs NO bids may indicate informed trader positioning.
- **Why it might work:** Informed traders (who see Binance price closer to settlement) would position in the final minutes. This is late-arriving information that our model doesn't see.
- **Data needed:** Kalshi WebSocket orderbook data (we currently only poll REST)
- **Effort:** Medium-High (need WebSocket integration for Kalshi orderbook)
- **Risks:** Kalshi 15-min market orderbooks may be too thin for meaningful skew signals.

### 6. GARCH Volatility Regime Enhancement
- **What:** Replace our simple trend_score regime detector with a GARCH(1,1) volatility model. Use predicted volatility to adjust position sizing and signal thresholds.
- **Why it might work:** GARCH captures volatility clustering better than our current approach. High-volatility regimes may need wider thresholds; low-vol may allow tighter ones.
- **Data needed:** Already have (1-min candles)
- **Effort:** Medium (1-2 sessions)
- **Risks:** GARCH may not add much over our current regime detector for 15-min windows. Worth backtesting first.

### 7. Funding Rate Feature
- **What:** Add Binance perpetual futures funding rate as an XGBoost feature. Extreme positive funding = long crowding = reversal risk.
- **Why it might work:** Funding rates update every 8 hours but the *level* of current funding indicates leverage positioning. Extreme funding is a contrarian signal.
- **Data needed:** Binance API (free, already connected)
- **Effort:** Low (add one feature to XGBoost input)
- **Risks:** Funding updates are infrequent (8h) vs our 15-min window. May be too slow to matter.

---

## D. What NOT to Pursue

| Idea | Why Skip |
|---|---|
| LLM consensus trading | Too slow for 15-min crypto; $10/day API cost; no evidence of edge on price-action markets |
| Cross-platform arbitrage | Spreads compressed to 1-2% in 2026; needs microsecond infra; not viable at our capital |
| Market making | Needs $10k+ capital and 20+ simultaneous positions; our $100-500 bankroll is too small |
| On-chain signals | Too slow for 15-min scale; whale moves predict hours/days, not minutes |
| Fear & Greed Index | Updates too slowly; better for daily/weekly timeframes |
| Social media sentiment NLP | Latency too high for 15-min windows; academic accuracy ~55% max |

---

## E. Questions for the User

1. **Current live performance:** What's your win rate and P&L over the last few days since retraining? The 59.7% validation WR needs live confirmation.

2. **Binance data granularity:** Are you receiving the Binance `forceOrder` (liquidation) stream, or just OHLCV candles + order book depth? If not, would you be open to adding it?

3. **Kalshi WebSocket:** Are you using Kalshi's WebSocket for market data, or only REST polling? WebSocket would enable orderbook skew analysis and latency measurement.

4. **Capital timeline:** Are you planning to stay at $100-500 bankroll for a while, or do you want to scale up soon? Some strategies (market making, arbitrage) only make sense at higher capital.

5. **Risk appetite for new signals:** Would you prefer to add new signals as heuristic rules (fast, easy to test/disable) or as XGBoost features (requires retrain, harder to isolate effect)?

6. **Contract price data:** How much data has the KalshiPriceLogger accumulated so far? This determines whether latency analysis is feasible now.

7. **Time commitment:** How much time per week can you dedicate to bot development? This affects whether we pursue 1 idea deeply or 3 ideas quickly.

---

## F. Recommended Implementation Order

If we proceed, I'd suggest this sequence:

1. **Latency analysis** (1 session, low effort) — Analyze existing price logger data to measure Binance-to-Kalshi lag. If no lag exists, skip the latency arb idea entirely. If lag exists, it's probably the easiest alpha to capture.

2. **Previous-market autocorrelation test** (30 min) — Quick statistical test on settlement history. If autocorrelation exists, add as heuristic rule.

3. **Jump/spike mean-reversion heuristic** (1 session) — Backtest on historical data, then add as heuristic rule if positive.

4. **Liquidation cascade signal** (1-2 sessions) — Add Binance liquidation stream, engineer features, backtest, potentially retrain XGBoost.

5. **Funding rate feature** (30 min) — Easy to add as an XGBoost feature during next retrain.

6. **GARCH regime detector** (1-2 sessions) — Replace current regime detector if backtesting shows improvement.

7. **Orderbook skew** (2-3 sessions) — Only if Kalshi WebSocket integration is worthwhile for other reasons too.

---

## G. Repo/Resource Bookmarks

**Most relevant repos to revisit:**
- [simple-kalshi-bot](https://github.com/DeweyMarco/simple-kalshi-bot) — KXBTC15M strategies, closest to our use case
- [polymarket-latency-bot](https://github.com/learningworship/polymarket-latency-bot) — latency arb approach
- [Poly-Tutor bots](https://github.com/Poly-Tutor/polymarket-5min-15min-1hour-arbitrage-trading-bot-tools) — VWAP + orderbook skew
- [kalshi-market-making](https://github.com/nikhilnd/kalshi-market-making) — Cauchy distribution pricing (reference)
- [pykalshi](https://github.com/arshka/pykalshi) — if we need WebSocket support
- [prediction-market-analysis](https://github.com/Jon-Becker/prediction-market-analysis) — largest public Kalshi dataset

**Key academic sources:**
- [Makers and Takers: Economics of Kalshi](https://www.karlwhelan.com/Papers/Kalshi.pdf) — the -20% average return study
- [Toward Black-Scholes for Prediction Markets](https://arxiv.org/pdf/2510.15205) — pricing theory
- [Short-Term Bitcoin Prediction via ML](https://www.sciencedirect.com/science/article/pii/S2405918821000027) — 50.9-56% accuracy ceiling
