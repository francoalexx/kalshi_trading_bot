"""
Paper trading P&L tracker.

Monitors open paper trades and checks their outcomes once the Kalshi market
settles. Prints a live dashboard to the console showing:
  - Win/Loss record
  - Net P&L (simulated)
  - Win rate
  - Per-trade log
"""

import time
from dataclasses import dataclass, field
from typing import List, Optional

from loguru import logger

from src.notifications import send_settlement_email


@dataclass
class PaperTrade:
    trade_id:     int
    ticker:       str
    side:         str        # "yes" or "no"
    contracts:    int
    entry_price:  float      # 0-1 scale
    expiry_ts:    float      # unix epoch
    signal_edge:  float      # net_edge from alpha engine
    settled:      bool       = False
    won:          bool       = False
    pnl:          float      = 0.0
    fee:          float      = 0.0


class PaperTracker:
    """
    Tracks simulated trades and computes P&L when markets expire.

    How settlement works:
      - Each Kalshi contract pays $1.00 if you win, $0.00 if you lose.
      - If you bought YES at $0.55 and BTC went up → win: profit = $1.00 - $0.55 = $0.45/contract
      - If you bought YES at $0.55 and BTC went down → loss: -$0.55/contract
      - Fee = 0.07 × p × (1-p) per contract (Kalshi taker formula)
    """

    def __init__(self, starting_bankroll: float = 500.0):
        self._bankroll  = starting_bankroll
        self._start     = starting_bankroll
        self._trades: List[PaperTrade] = []
        self._counter   = 0

    def record_trade(
        self,
        ticker: str,
        side: str,
        contracts: int,
        entry_price: float,
        expiry_ts: float,
        signal_edge: float,
    ) -> PaperTrade:
        self._counter += 1
        trade = PaperTrade(
            trade_id=self._counter,
            ticker=ticker,
            side=side,
            contracts=contracts,
            entry_price=entry_price,
            expiry_ts=expiry_ts,
            signal_edge=signal_edge,
        )
        self._trades.append(trade)
        logger.info(
            f"[PAPER] Trade #{self._counter} recorded: "
            f"{side.upper()} {contracts}x {ticker} @ {entry_price:.3f}"
        )
        return trade

    def settle_by_result(self, trade: PaperTrade, market_result: str) -> None:
        """
        Settle a paper trade given the Kalshi market result string.

        Parameters
        ----------
        trade         : the PaperTrade to settle
        market_result : "yes" or "no" — the side that won on Kalshi
        """
        won = (trade.side == market_result)
        self._finish_settlement(trade, won)

    def settle_trade(self, trade: PaperTrade, btc_went_up: bool) -> None:
        """
        Settle a paper trade given the actual BTC outcome.

        Parameters
        ----------
        trade       : the PaperTrade to settle
        btc_went_up : True if BTC closed higher at market expiry
        """
        won = (trade.side == "yes" and btc_went_up) or \
              (trade.side == "no"  and not btc_went_up)
        self._finish_settlement(trade, won)

    def _finish_settlement(self, trade: PaperTrade, won: bool) -> None:
        """Common settlement logic shared by both settle methods."""

        if trade.settled:
            return
        if won:
            gross_pnl = (1.0 - trade.entry_price) * trade.contracts
        else:
            gross_pnl = -trade.entry_price * trade.contracts

        fee = 0.07 * trade.entry_price * (1 - trade.entry_price) * trade.contracts
        net_pnl = gross_pnl - fee

        trade.settled = True
        trade.won     = won
        trade.pnl     = net_pnl
        trade.fee     = fee
        self._bankroll += net_pnl

        result = "WIN ✓" if won else "LOSS ✗"
        logger.info(
            f"[PAPER] Trade #{trade.trade_id} settled — {result} | "
            f"pnl={net_pnl:+.2f} fee={fee:.2f} | "
            f"bankroll=${self._bankroll:.2f}"
        )
        self.print_dashboard()
        send_settlement_email(
            trade=trade,
            bankroll=self._bankroll,
            start_bankroll=self._start,
            settled_trades=[t for t in self._trades if t.settled],
        )

    def print_dashboard(self) -> None:
        """Print current paper trading performance to console."""
        settled = [t for t in self._trades if t.settled]
        open_   = [t for t in self._trades if not t.settled]

        if not settled:
            print("\n── Paper Trading Dashboard ──────────────────")
            print(f"  No settled trades yet. Open: {len(open_)}")
            print("─────────────────────────────────────────────\n")
            return

        wins      = sum(1 for t in settled if t.won)
        losses    = len(settled) - wins
        win_rate  = wins / len(settled) if settled else 0
        total_pnl = sum(t.pnl for t in settled)
        total_fee = sum(t.fee for t in settled)
        roi       = (self._bankroll - self._start) / self._start * 100

        print("\n── Paper Trading Dashboard ──────────────────")
        print(f"  Bankroll : ${self._bankroll:.2f}  (started ${self._start:.2f})")
        print(f"  ROI      : {roi:+.2f}%")
        print(f"  Net P&L  : ${total_pnl:+.2f}  (fees: ${total_fee:.2f})")
        print(f"  Record   : {wins}W / {losses}L  ({win_rate:.1%} win rate)")
        print(f"  Open     : {len(open_)} trades pending")
        print(f"  Trades   : {len(settled)} settled")
        print("─────────────────────────────────────────────")
        print(f"  {'#':>3}  {'Side':<5}  {'Contracts':>9}  {'Entry':>6}  {'P&L':>8}  {'Result'}")
        print(f"  {'─'*3}  {'─'*5}  {'─'*9}  {'─'*6}  {'─'*8}  {'─'*6}")
        for t in settled[-10:]:   # last 10 trades
            res = "WIN" if t.won else "LOSS"
            print(
                f"  {t.trade_id:>3}  {t.side.upper():<5}  "
                f"{t.contracts:>9}  {t.entry_price:>6.3f}  "
                f"{t.pnl:>+8.2f}  {res}"
            )
        print("─────────────────────────────────────────────\n")
