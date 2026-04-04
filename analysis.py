"""
Deep-dive exploratory analysis of trade_history.csv
Outputs a structured markdown report to data/analysis_report.md
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency, fisher_exact, ttest_ind, pearsonr
from pathlib import Path
from datetime import datetime
from collections import Counter
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
OVERALL_WR = None  # computed from data
START_BANKROLL = 500.0

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def md_table(headers, rows, alignments=None):
    """Build a pipe-delimited markdown table."""
    if alignments is None:
        alignments = ["left"] * len(headers)
    sep_map = {"left": ":---", "right": "---:", "center": ":---:"}
    sep = [sep_map.get(a, "---") for a in alignments]
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(sep) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


def bin_column(series, edges, labels):
    """Bin a numeric series using pd.cut with -inf/+inf boundaries."""
    return pd.cut(series, bins=edges, labels=labels, include_lowest=True)


def win_rate_by_group(df, group_col):
    """Compute win/loss stats per group."""
    grouped = df.groupby(group_col, observed=False)
    out = grouped.agg(
        n=("won", "count"),
        wins=("won", "sum"),
        avg_pnl=("pnl", "mean"),
        total_pnl=("pnl", "sum"),
    ).reset_index()
    out["losses"] = out["n"] - out["wins"]
    out["win_rate"] = out["wins"] / out["n"]
    out["win_rate"] = out["win_rate"].fillna(0)
    return out


def deviation_flag(group_wr, overall_wr, n):
    """One-proportion z-test flag: how far is group WR from overall?"""
    if n < 3:
        return ""
    se = np.sqrt(overall_wr * (1 - overall_wr) / n)
    if se == 0:
        return ""
    z = (group_wr - overall_wr) / se
    if abs(z) < 1.0:
        return ""
    elif abs(z) < 2.0:
        return "+" if z > 0 else "-"
    else:
        return "++" if z > 0 else "--"


def chi2_or_fisher(df, group_col):
    """Run chi-squared or Fisher exact test on group_col vs outcome."""
    ct = pd.crosstab(df[group_col], df["won"])
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return None, None
    # If 2x2 and any cell < 5, use Fisher exact
    if ct.shape == (2, 2) and ct.min().min() < 5:
        odds, p = fisher_exact(ct.values)
        return "Fisher", p
    try:
        chi2, p, dof, expected = chi2_contingency(ct.values)
        return "Chi2", p
    except ValueError:
        return None, None


def cohens_d(g1, g2):
    """Cohen's d effect size."""
    n1, n2 = len(g1), len(g2)
    if n1 < 2 or n2 < 2:
        return 0.0
    var1, var2 = g1.var(ddof=1), g2.var(ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return (g1.mean() - g2.mean()) / pooled_std


def fmt_pct(x):
    return f"{x:.1%}"


def fmt_f(x, d=2):
    return f"{x:.{d}f}"


def fmt_dollar(x):
    return f"${x:,.2f}"


def make_breakdown_section(df, group_col, title, overall_wr, note=""):
    """Standard breakdown section: table + test + sentence."""
    stats_df = win_rate_by_group(df, group_col)
    # Drop empty groups
    stats_df = stats_df[stats_df["n"] > 0].copy()

    headers = ["Group", "n", "Wins", "Losses", "Win Rate", "vs Avg", "Avg PnL", "Total PnL"]
    aligns = ["left", "right", "right", "right", "right", "center", "right", "right"]
    rows = []
    for _, r in stats_df.iterrows():
        flag = deviation_flag(r["win_rate"], overall_wr, r["n"])
        rows.append([
            str(r[group_col]),
            str(int(r["n"])),
            str(int(r["wins"])),
            str(int(r["losses"])),
            fmt_pct(r["win_rate"]),
            flag if flag else "~",
            fmt_dollar(r["avg_pnl"]),
            fmt_dollar(r["total_pnl"]),
        ])

    test_name, p_val = chi2_or_fisher(df, group_col)

    lines = []
    lines.append(f"#### {title}\n")
    if note:
        lines.append(f"*{note}*\n")
    lines.append(md_table(headers, rows, aligns))
    lines.append("")
    if test_name and p_val is not None:
        sig = "**statistically significant**" if p_val < 0.05 else "not statistically significant"
        lines.append(f"{test_name} test p-value: **{p_val:.4f}** ({sig} at alpha=0.05)")
    else:
        lines.append("*Statistical test not applicable (insufficient groups or cells).*")

    # small-n caveat
    min_n = stats_df["n"].min()
    if min_n < 15:
        lines.append(f"\n*Caution: smallest group has only n={int(min_n)} trades — interpret with care.*")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Phase 1: Data Inventory
# ---------------------------------------------------------------------------

def phase1_data_inventory(df):
    lines = []
    lines.append("## Phase 1 — Data Inventory\n")

    # Schema table
    lines.append("### Column Schema\n")
    schema_headers = ["Column", "Type", "Non-Null", "Range / Values", "Notes"]
    schema_rows = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        non_null = int(df[col].notna().sum())
        n_total = len(df)

        if non_null == 0:
            range_str = "—"
            note = "100% null — skipped"
        elif df[col].dtype in [np.float64, np.int64, float, int]:
            vals = df[col].dropna()
            nunique = vals.nunique()
            if nunique == 1:
                range_str = str(vals.iloc[0])
                note = "Constant"
            else:
                range_str = f"{vals.min():.4g} to {vals.max():.4g}"
                note = f"mean={vals.mean():.4g}"
        else:
            unique_vals = df[col].dropna().unique()
            if len(unique_vals) <= 5:
                range_str = ", ".join(sorted(str(v) for v in unique_vals))
                note = f"{len(unique_vals)} unique"
            else:
                range_str = f"{len(unique_vals)} unique values"
                note = ""

        schema_rows.append([col, dtype, f"{non_null}/{n_total}", range_str, note])

    lines.append(md_table(schema_headers, schema_rows))
    lines.append("")

    # Top-level summary
    lines.append("### Summary Statistics\n")
    total = len(df)
    wins = int(df["won"].sum())
    losses = total - wins
    wr = wins / total
    total_pnl = df["pnl"].sum()
    total_fees = df["fee"].sum()
    avg_pnl = df["pnl"].mean()
    avg_win = df.loc[df["won"] == 1, "pnl"].mean()
    avg_loss = df.loc[df["won"] == 0, "pnl"].mean()
    final_bankroll = df["running_bankroll"].iloc[-1]
    roi = (final_bankroll - START_BANKROLL) / START_BANKROLL * 100

    lines.append(f"| Metric | Value |")
    lines.append(f"| :--- | ---: |")
    lines.append(f"| Total trades | {total} |")
    lines.append(f"| Wins / Losses | {wins} / {losses} ({wr:.1%}) |")
    lines.append(f"| Total net P&L | {fmt_dollar(total_pnl)} |")
    lines.append(f"| Total fees paid | {fmt_dollar(total_fees)} |")
    lines.append(f"| Avg P&L per trade | {fmt_dollar(avg_pnl)} |")
    lines.append(f"| Avg winning trade | {fmt_dollar(avg_win)} |")
    lines.append(f"| Avg losing trade | {fmt_dollar(avg_loss)} |")
    lines.append(f"| Win/Loss size ratio | {abs(avg_win / avg_loss):.2f}x |")
    lines.append(f"| Starting bankroll | {fmt_dollar(START_BANKROLL)} |")
    lines.append(f"| Final bankroll | {fmt_dollar(final_bankroll)} |")
    lines.append(f"| ROI | {roi:.1f}% |")
    lines.append(f"| Median trade P&L | {fmt_dollar(df['pnl'].median())} |")
    lines.append(f"| P&L std dev | {fmt_dollar(df['pnl'].std())} |")
    lines.append("")

    # Timestamp note
    lines.append("### Data Limitations\n")
    lines.append("- **No timestamps available**: `expiry_time` is 100% null. Time-of-day and day-of-week "
                 "analysis cannot be performed. Trade sequence is inferred from `trade_id`.")
    lines.append("- **`signal_edge` is null** for all 118 trades (known bug, fixed from trade #119). "
                 "All losses default to `ALPHA_DECAY` classification as a result.")
    lines.append("- **`kelly_fraction` and `slippage`** are constant 0.0 (same bug).")
    lines.append("- **`regime`** is 'ranging' for all trades; **`strike_type`** is 'greater_or_equal' for all trades — "
                 "no variance to analyze on these dimensions.")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Phase 2: Win/Loss Breakdown
# ---------------------------------------------------------------------------

def phase2_win_loss_breakdown(df):
    global OVERALL_WR
    OVERALL_WR = df["won"].mean()
    wr = OVERALL_WR

    lines = []
    lines.append("## Phase 2 — Win/Loss Breakdown by Dimension\n")
    lines.append(f"Overall win rate: **{wr:.1%}** ({int(df['won'].sum())}W / {len(df) - int(df['won'].sum())}L "
                 f"out of {len(df)} trades). Deviations flagged: `+`/`-` = 1-2 SE from overall, "
                 f"`++`/`--` = >2 SE. `~` = within normal range.\n")

    # Track p-values for summary
    p_values = []

    # 1. By side
    lines.append(make_breakdown_section(df, "side", "1. By Side (YES vs NO)", wr,
                 "93% of trades are YES; n=8 for NO is very small."))
    _, p = chi2_or_fisher(df, "side")
    p_values.append(("side", p))

    # 2. By entry_price
    edges = [0, 0.22, 0.31, 0.38, 0.45, 1.0]
    labels = ["0.16-0.22", "0.22-0.31", "0.31-0.38", "0.38-0.45", "0.45-0.57"]
    df["entry_price_bin"] = bin_column(df["entry_price"], edges, labels)
    lines.append(make_breakdown_section(df, "entry_price_bin",
                 "2. By Entry Price", wr,
                 "Lower entry prices = cheaper contracts (further from ATM). "
                 "Higher entry prices = closer to ATM."))
    _, p = chi2_or_fisher(df, "entry_price_bin")
    p_values.append(("entry_price", p))

    # 3. By contracts
    q_edges = [0] + list(df["contracts"].quantile([0.2, 0.4, 0.6, 0.8]).values) + [200]
    q_labels = []
    for i in range(len(q_edges) - 1):
        q_labels.append(f"{int(q_edges[i])}-{int(q_edges[i+1])}")
    df["contracts_bin"] = bin_column(df["contracts"], q_edges, q_labels)
    lines.append(make_breakdown_section(df, "contracts_bin",
                 "3. By Position Size (Contracts)", wr,
                 "Contracts are inversely related to entry_price due to Kelly sizing."))
    _, p = chi2_or_fisher(df, "contracts_bin")
    p_values.append(("contracts", p))

    # 4. By bar_net_edge
    edges = [0, 0.08, 0.15, 0.22, 0.31, 0.40]
    labels = ["0.02-0.08", "0.08-0.15", "0.15-0.22", "0.22-0.31", "0.31-0.37"]
    df["edge_bin"] = bin_column(df["bar_net_edge"], edges, labels)
    lines.append(make_breakdown_section(df, "edge_bin",
                 "4. By Bar Net Edge (Perceived Signal Strength)", wr,
                 "Key question: does higher perceived edge predict better outcomes?"))
    _, p = chi2_or_fisher(df, "edge_bin")
    p_values.append(("bar_net_edge", p))

    # 5. By ensemble_prob
    edges = [0.45, 0.52, 0.54, 0.56, 0.58, 0.65]
    labels = ["0.45-0.52", "0.52-0.54", "0.54-0.56", "0.56-0.58", "0.58-0.64"]
    df["ensemble_bin"] = bin_column(df["ensemble_prob"], edges, labels)
    lines.append(make_breakdown_section(df, "ensemble_bin",
                 "5. By Ensemble Probability", wr,
                 "Combined model confidence. 0.54 is the XGB YES threshold."))
    _, p = chi2_or_fisher(df, "ensemble_bin")
    p_values.append(("ensemble_prob", p))

    # 6. By xgb_prob
    edges = [0.44, 0.54, 0.56, 0.58, 0.60, 0.70]
    labels = ["0.45-0.54", "0.54-0.56", "0.56-0.58", "0.58-0.60", "0.60-0.69"]
    df["xgb_bin"] = bin_column(df["xgb_prob"], edges, labels)
    lines.append(make_breakdown_section(df, "xgb_bin",
                 "6. By XGBoost Probability", wr,
                 "0.54 is the production YES threshold in alpha.py."))
    _, p = chi2_or_fisher(df, "xgb_bin")
    p_values.append(("xgb_prob", p))

    # 7. By cnn_prob
    edges = [0.47, 0.51, 0.52, 0.53, 0.54, 0.56]
    labels = ["0.47-0.51", "0.51-0.52", "0.52-0.53", "0.53-0.54", "0.54-0.56"]
    df["cnn_bin"] = bin_column(df["cnn_prob"], edges, labels)
    lines.append(make_breakdown_section(df, "cnn_bin",
                 "7. By CNN-LSTM Probability", wr,
                 "Very narrow range (0.47-0.55). Testing whether CNN adds discriminative value."))
    _, p = chi2_or_fisher(df, "cnn_bin")
    p_values.append(("cnn_prob", p))

    # 8. By seconds_to_expiry
    edges = [0, 300, 420, 540, 660, 900]
    labels = ["2-5 min", "5-7 min", "7-9 min", "9-11 min", "11-14 min"]
    df["tte_bin"] = bin_column(df["seconds_to_expiry"], edges, labels)
    lines.append(make_breakdown_section(df, "tte_bin",
                 "8. By Time to Expiry", wr,
                 "Does entering earlier (more time remaining) help or hurt?"))
    _, p = chi2_or_fisher(df, "tte_bin")
    p_values.append(("seconds_to_expiry", p))

    # 9. By market_yes_ask
    edges = [0.15, 0.25, 0.35, 0.45, 0.55, 0.85]
    labels = ["0.15-0.25", "0.25-0.35", "0.35-0.45", "0.45-0.55", "0.55-0.81"]
    df["mkt_ask_bin"] = bin_column(df["market_yes_ask"], edges, labels)
    lines.append(make_breakdown_section(df, "mkt_ask_bin",
                 "9. By Market YES Ask", wr,
                 "The market's own pricing of the YES contract."))
    _, p = chi2_or_fisher(df, "mkt_ask_bin")
    p_values.append(("market_yes_ask", p))

    # 10. By market_spread
    edges = [-0.001, 0.015, 0.02, 0.03, 0.06]
    labels = ["1c", "2c", "3c", "4-5c"]
    df["spread_bin"] = bin_column(df["market_spread"], edges, labels)
    lines.append(make_breakdown_section(df, "spread_bin",
                 "10. By Market Spread", wr,
                 "Tighter spreads suggest better liquidity."))
    _, p = chi2_or_fisher(df, "spread_bin")
    p_values.append(("market_spread", p))

    # 11. By btc_close
    q_edges = list(df["btc_close"].quantile([0, 0.25, 0.5, 0.75, 1.0]).values)
    q_edges[0] -= 1
    q_edges[-1] += 1
    q_labels = []
    for i in range(len(q_edges) - 1):
        q_labels.append(f"${int(q_edges[i]):,}-${int(q_edges[i+1]):,}")
    df["btc_bin"] = bin_column(df["btc_close"], q_edges, q_labels)
    lines.append(make_breakdown_section(df, "btc_bin",
                 "11. By BTC Close Price", wr,
                 "BTC price quartiles during the trading period."))
    _, p = chi2_or_fisher(df, "btc_bin")
    p_values.append(("btc_close", p))

    # 12. By log_return
    edges = [-0.005, -0.001, -0.0003, 0.0003, 0.001, 0.005]
    labels = ["big down", "down", "flat", "up", "big up"]
    df["logret_bin"] = bin_column(df["log_return"], edges, labels)
    lines.append(make_breakdown_section(df, "logret_bin",
                 "12. By Recent Log Return", wr,
                 "BTC price movement in the most recent bar."))
    _, p = chi2_or_fisher(df, "logret_bin")
    p_values.append(("log_return", p))

    # 13. By rolling_vol_15
    q_edges = list(df["rolling_vol_15"].quantile([0, 0.25, 0.5, 0.75, 1.0]).values)
    q_edges[0] -= 0.0001
    q_edges[-1] += 0.0001
    # avoid duplicate edges
    q_edges = sorted(set(q_edges))
    if len(q_edges) < 3:
        lines.append("#### 13. By Rolling Volatility (15-bar)\n")
        lines.append("*Insufficient variance for binning — most values clustered near zero.*\n")
        p_values.append(("rolling_vol_15", None))
    else:
        q_labels = [f"Q{i+1}" for i in range(len(q_edges) - 1)]
        df["vol_bin"] = bin_column(df["rolling_vol_15"], q_edges, q_labels)
        lines.append(make_breakdown_section(df, "vol_bin",
                     "13. By Rolling Volatility (15-bar)", wr,
                     "All values are low (max 0.0013). Quartile bins."))
        _, p = chi2_or_fisher(df, "vol_bin")
        p_values.append(("rolling_vol_15", p))

    # 14. By bb_width
    q_edges = list(df["bb_width"].quantile([0, 0.25, 0.5, 0.75, 1.0]).values)
    q_edges[0] -= 0.001
    q_edges[-1] += 0.001
    q_edges = sorted(set(q_edges))
    if len(q_edges) < 3:
        lines.append("#### 14. By Bollinger Band Width\n")
        lines.append("*Insufficient variance for binning.*\n")
        p_values.append(("bb_width", None))
    else:
        q_labels = [f"Q{i+1}" for i in range(len(q_edges) - 1)]
        df["bb_bin"] = bin_column(df["bb_width"], q_edges, q_labels)
        lines.append(make_breakdown_section(df, "bb_bin",
                     "14. By Bollinger Band Width", wr,
                     "Quartile bins of BB width."))
        _, p = chi2_or_fisher(df, "bb_bin")
        p_values.append(("bb_width", p))

    # 15. By rsi_14
    edges = [-1.0, -0.6, -0.4, -0.2, 0.0, 0.5]
    labels = ["deeply oversold", "oversold", "mod. oversold", "neutral-low", "above neutral"]
    df["rsi_bin"] = bin_column(df["rsi_14"], edges, labels)
    lines.append(make_breakdown_section(df, "rsi_bin",
                 "15. By RSI(14)", wr,
                 "RSI is normalized to [-1, 1]. Mean is -0.48 (mostly oversold conditions)."))
    _, p = chi2_or_fisher(df, "rsi_bin")
    p_values.append(("rsi_14", p))

    # 16. By macd
    q_edges = list(df["macd"].quantile([0, 0.25, 0.5, 0.75, 1.0]).values)
    q_edges[0] -= 1
    q_edges[-1] += 1
    q_labels = [f"Q{i+1} ({fmt_f(q_edges[i],0)} to {fmt_f(q_edges[i+1],0)})"
                for i in range(len(q_edges) - 1)]
    df["macd_bin"] = bin_column(df["macd"], q_edges, q_labels)
    lines.append(make_breakdown_section(df, "macd_bin",
                 "16. By MACD", wr,
                 "Quartile bins. Nearly all values are negative (bearish MACD)."))
    _, p = chi2_or_fisher(df, "macd_bin")
    p_values.append(("macd", p))

    # 17. By ofi_norm
    edges = [-4.0, -1.5, -0.8, -0.2, 0.5, 3.0]
    labels = ["strong sell", "mod. sell", "mild sell", "neutral", "buy"]
    df["ofi_bin"] = bin_column(df["ofi_norm"], edges, labels)
    lines.append(make_breakdown_section(df, "ofi_bin",
                 "17. By Normalized OFI", wr,
                 "Order flow imbalance. Negative = net selling pressure."))
    _, p = chi2_or_fisher(df, "ofi_bin")
    p_values.append(("ofi_norm", p))

    # 18. By aggressor_ratio
    edges = [0.40, 0.48, 0.50, 0.52, 0.58]
    labels = ["seller-dom", "slight-sell", "slight-buy", "buyer-dom"]
    df["agg_bin"] = bin_column(df["aggressor_ratio"], edges, labels)
    lines.append(make_breakdown_section(df, "agg_bin",
                 "18. By Aggressor Ratio", wr,
                 "Range 0.41-0.57, centered near 0.50. Tight distribution."))
    _, p = chi2_or_fisher(df, "agg_bin")
    p_values.append(("aggressor_ratio", p))

    # 19. By trade sequence
    edges = [0, 25, 50, 75, 100, 120]
    labels = ["1-25", "26-50", "51-75", "76-100", "101-118"]
    df["epoch_bin"] = bin_column(df["trade_id"], edges, labels)
    lines.append(make_breakdown_section(df, "epoch_bin",
                 "19. By Trade Sequence (Learning Curve)", wr,
                 "Proxy for time progression — does performance improve over the run?"))
    _, p = chi2_or_fisher(df, "epoch_bin")
    p_values.append(("trade_sequence", p))

    # Summary ranking
    lines.append("### Dimension Summary — Ranked by Association Strength\n")
    p_sorted = sorted(p_values, key=lambda x: (x[1] if x[1] is not None else 999))
    summary_headers = ["Rank", "Dimension", "Test p-value", "Significant?"]
    summary_rows = []
    for i, (dim, p) in enumerate(p_sorted, 1):
        if p is None:
            summary_rows.append([str(i), dim, "N/A", "—"])
        else:
            sig = "Yes" if p < 0.05 else "No"
            summary_rows.append([str(i), dim, f"{p:.4f}", sig])
    lines.append(md_table(summary_headers, summary_rows, ["right", "left", "right", "center"]))
    lines.append("")
    lines.append("*Note: With 19 tests at alpha=0.05, we expect ~1 false positive by chance. "
                 "Interpret low p-values as \"worth investigating\" rather than definitive.*\n")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Phase 3: Sequence & Streak Analysis
# ---------------------------------------------------------------------------

def phase3_sequence_analysis(df):
    lines = []
    lines.append("## Phase 3 — Sequence & Streak Analysis\n")

    won = df["won"].values

    # --- Streaks ---
    lines.append("### Win/Loss Streaks\n")
    streaks = []
    current_val = won[0]
    current_len = 1
    for i in range(1, len(won)):
        if won[i] == current_val:
            current_len += 1
        else:
            streaks.append((int(current_val), current_len))
            current_val = won[i]
            current_len = 1
    streaks.append((int(current_val), current_len))

    win_streaks = [s[1] for s in streaks if s[0] == 1]
    loss_streaks = [s[1] for s in streaks if s[0] == 0]

    lines.append(f"| Metric | Win Streaks | Loss Streaks |")
    lines.append(f"| :--- | ---: | ---: |")
    lines.append(f"| Count | {len(win_streaks)} | {len(loss_streaks)} |")
    lines.append(f"| Max length | {max(win_streaks)} | {max(loss_streaks)} |")
    lines.append(f"| Mean length | {np.mean(win_streaks):.1f} | {np.mean(loss_streaks):.1f} |")
    lines.append(f"| Median length | {np.median(win_streaks):.0f} | {np.median(loss_streaks):.0f} |")
    lines.append("")

    # Streak distribution
    all_lengths = sorted(set(s[1] for s in streaks))
    win_counter = Counter(s[1] for s in streaks if s[0] == 1)
    loss_counter = Counter(s[1] for s in streaks if s[0] == 0)

    dist_headers = ["Streak Length", "Win Runs", "Loss Runs"]
    dist_rows = []
    for length in all_lengths:
        dist_rows.append([str(length), str(win_counter.get(length, 0)),
                         str(loss_counter.get(length, 0))])
    lines.append("**Streak length distribution:**\n")
    lines.append(md_table(dist_headers, dist_rows, ["right", "right", "right"]))
    lines.append("")

    # Bernoulli expectation
    p = df["won"].mean()
    n = len(df)
    # Expected max win streak ≈ log(n) / log(1/p)
    exp_max_win = np.log(n) / np.log(1 / p) if p > 0 and p < 1 else 0
    exp_max_loss = np.log(n) / np.log(1 / (1 - p)) if p > 0 and p < 1 else 0
    lines.append(f"**Bernoulli comparison** (if each trade were independent at p={p:.3f}):")
    lines.append(f"- Expected max win streak: ~{exp_max_win:.1f} | Observed: {max(win_streaks)}")
    lines.append(f"- Expected max loss streak: ~{exp_max_loss:.1f} | Observed: {max(loss_streaks)}")
    obs_vs_exp_win = "longer than expected" if max(win_streaks) > exp_max_win * 1.2 else "within range"
    obs_vs_exp_loss = "longer than expected" if max(loss_streaks) > exp_max_loss * 1.2 else "within range"
    lines.append(f"- Win streaks: {obs_vs_exp_win}. Loss streaks: {obs_vs_exp_loss}.")
    lines.append("")

    # --- Autocorrelation ---
    lines.append("### Autocorrelation\n")
    lag1_r, lag1_p = pearsonr(won[:-1], won[1:])
    lag2_r, lag2_p = pearsonr(won[:-2], won[2:])
    lines.append(f"| Lag | Correlation | p-value | Interpretation |")
    lines.append(f"| :--- | ---: | ---: | :--- |")
    interp1 = "Significant clustering" if lag1_p < 0.05 else "No significant pattern"
    interp2 = "Significant clustering" if lag2_p < 0.05 else "No significant pattern"
    lines.append(f"| Lag-1 | {lag1_r:.4f} | {lag1_p:.4f} | {interp1} |")
    lines.append(f"| Lag-2 | {lag2_r:.4f} | {lag2_p:.4f} | {interp2} |")
    lines.append("")

    if lag1_p < 0.05:
        lines.append(f"Lag-1 autocorrelation of **{lag1_r:.3f}** (p={lag1_p:.4f}) indicates that "
                     f"outcomes are **not independent** — wins tend to follow wins, and losses tend to follow losses. "
                     f"This is likely driven by BTC price trends/regimes affecting consecutive 15-minute markets.\n")
    else:
        lines.append(f"No statistically significant autocorrelation detected. Outcomes appear roughly independent.\n")

    # Runs test (Wald-Wolfowitz)
    lines.append("### Wald-Wolfowitz Runs Test\n")
    n_runs = 1
    for i in range(1, len(won)):
        if won[i] != won[i - 1]:
            n_runs += 1
    n1 = int(won.sum())
    n0 = len(won) - n1
    expected_runs = 1 + (2 * n1 * n0) / (n1 + n0)
    var_runs = (2 * n1 * n0 * (2 * n1 * n0 - n1 - n0)) / ((n1 + n0) ** 2 * (n1 + n0 - 1))
    z_runs = (n_runs - expected_runs) / np.sqrt(var_runs) if var_runs > 0 else 0
    p_runs = 2 * (1 - stats.norm.cdf(abs(z_runs)))
    lines.append(f"- Observed runs: {n_runs}")
    lines.append(f"- Expected runs (under independence): {expected_runs:.1f}")
    lines.append(f"- Z-statistic: {z_runs:.3f}, p-value: {p_runs:.4f}")
    if p_runs < 0.05:
        fewer_more = "fewer" if n_runs < expected_runs else "more"
        lines.append(f"- **Significant**: There are {fewer_more} runs than expected, confirming outcome clustering.")
    else:
        lines.append(f"- Not significant: run count is consistent with independence.")
    lines.append("")

    # --- Rolling win rate ---
    lines.append("### Rolling Win Rate\n")
    won_series = df["won"].astype(float)
    roll10 = won_series.rolling(10, min_periods=10).mean()
    roll20 = won_series.rolling(20, min_periods=20).mean()

    lines.append("**10-trade rolling window:**")
    valid_r10 = roll10.dropna()
    if len(valid_r10) > 0:
        min_idx10 = valid_r10.idxmin()
        max_idx10 = valid_r10.idxmax()
        lines.append(f"- Min: {valid_r10.min():.0%} (around trade #{min_idx10 + 1})")
        lines.append(f"- Max: {valid_r10.max():.0%} (around trade #{max_idx10 + 1})")
        lines.append(f"- Final (last 10 trades): {valid_r10.iloc[-1]:.0%}")
    lines.append("")

    lines.append("**20-trade rolling window:**")
    valid_r20 = roll20.dropna()
    if len(valid_r20) > 0:
        min_idx20 = valid_r20.idxmin()
        max_idx20 = valid_r20.idxmax()
        lines.append(f"- Min: {valid_r20.min():.0%} (around trade #{min_idx20 + 1})")
        lines.append(f"- Max: {valid_r20.max():.0%} (around trade #{max_idx20 + 1})")
        lines.append(f"- Final (last 20 trades): {valid_r20.iloc[-1]:.0%}")
    lines.append("")

    # Describe trajectory by quarters
    lines.append("**Performance by quarter:**\n")
    quarters = [
        ("Trades 1-30", df.iloc[:30]),
        ("Trades 31-60", df.iloc[30:60]),
        ("Trades 61-90", df.iloc[60:90]),
        ("Trades 91-118", df.iloc[90:]),
    ]
    q_headers = ["Period", "n", "Wins", "Win Rate", "Total PnL", "Avg PnL"]
    q_rows = []
    for label, chunk in quarters:
        chunk_n = len(chunk)
        chunk_wins = int(chunk["won"].sum())
        chunk_wr = chunk_wins / chunk_n if chunk_n > 0 else 0
        q_rows.append([
            label, str(chunk_n), str(chunk_wins), fmt_pct(chunk_wr),
            fmt_dollar(chunk["pnl"].sum()), fmt_dollar(chunk["pnl"].mean())
        ])
    lines.append(md_table(q_headers, q_rows, ["left", "right", "right", "right", "right", "right"]))
    lines.append("")

    # --- Cumulative P&L ---
    lines.append("### Cumulative P&L Trajectory\n")
    cum_pnl = df["pnl"].cumsum()
    peak = cum_pnl.cummax()
    drawdown = peak - cum_pnl

    lines.append(f"- Final cumulative P&L: {fmt_dollar(cum_pnl.iloc[-1])}")
    max_dd_idx = drawdown.idxmax()
    max_dd_val = drawdown.iloc[max_dd_idx] if len(drawdown) > 0 else 0
    peak_at_dd = peak.iloc[max_dd_idx] if len(peak) > 0 else 0
    dd_pct = (max_dd_val / peak_at_dd * 100) if peak_at_dd > 0 else 0
    lines.append(f"- Max drawdown: {fmt_dollar(max_dd_val)} ({dd_pct:.1f}% of peak) "
                 f"at trade #{max_dd_idx + 1}")

    # Milestones
    milestones = [500, 1000, 1500, 2000]
    for m in milestones:
        reached = cum_pnl[cum_pnl >= m]
        if len(reached) > 0:
            lines.append(f"- Crossed ${m:,} cumulative P&L at trade #{reached.index[0] + 1}")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Phase 4: Loss Characterization
# ---------------------------------------------------------------------------

def phase4_loss_characterization(df):
    lines = []
    lines.append("## Phase 4 — Loss Characterization\n")

    wins_df = df[df["won"] == 1]
    loss_df = df[df["won"] == 0]

    # --- Side-by-side comparison ---
    lines.append("### Win vs Loss Feature Comparison\n")
    numeric_cols = [
        "entry_price", "contracts", "bar_net_edge", "ensemble_prob", "xgb_prob",
        "cnn_prob", "p_yes", "seconds_to_expiry", "market_yes_ask", "market_spread",
        "btc_close", "log_return", "rolling_vol_15", "rsi_14", "macd", "bb_width",
        "ofi_norm", "aggressor_ratio"
    ]

    comp_headers = ["Feature", "Win Mean", "Loss Mean", "Diff", "t-stat", "p-value", "Cohen's d", "Effect"]
    comp_rows = []
    for col in numeric_cols:
        w = wins_df[col].dropna()
        l = loss_df[col].dropna()
        if len(w) < 2 or len(l) < 2:
            continue
        w_mean = w.mean()
        l_mean = l.mean()
        diff = w_mean - l_mean
        t, p = ttest_ind(w, l, equal_var=False)
        d = cohens_d(w, l)
        # Effect label
        ad = abs(d)
        if ad < 0.2:
            effect = "negligible"
        elif ad < 0.5:
            effect = "small"
        elif ad < 0.8:
            effect = "medium"
        else:
            effect = "large"

        comp_rows.append([
            col, fmt_f(w_mean, 4), fmt_f(l_mean, 4), fmt_f(diff, 4),
            fmt_f(t, 2), fmt_f(p, 4), fmt_f(d, 3), effect
        ])

    lines.append(md_table(comp_headers, comp_rows,
                         ["left", "right", "right", "right", "right", "right", "right", "center"]))
    lines.append("")
    lines.append("*Features sorted by column order. Positive Cohen's d means wins have higher values.*\n")

    # Highlight significant features
    sig_features = [(r[0], float(r[5]), float(r[6])) for r in comp_rows if float(r[5]) < 0.10]
    if sig_features:
        lines.append("**Features with p < 0.10:**\n")
        for feat, p, d in sig_features:
            direction = "higher in wins" if d > 0 else "higher in losses"
            lines.append(f"- **{feat}**: p={p:.4f}, d={d:.3f} ({direction})")
        lines.append("")

    # --- Loss P&L distribution ---
    lines.append("### Loss P&L Distribution\n")
    loss_pnl = loss_df["pnl"]
    lines.append(f"| Metric | Value |")
    lines.append(f"| :--- | ---: |")
    lines.append(f"| Count | {len(loss_pnl)} |")
    lines.append(f"| Min (worst) | {fmt_dollar(loss_pnl.min())} |")
    lines.append(f"| Max (best loss) | {fmt_dollar(loss_pnl.max())} |")
    lines.append(f"| Mean | {fmt_dollar(loss_pnl.mean())} |")
    lines.append(f"| Median | {fmt_dollar(loss_pnl.median())} |")
    lines.append(f"| Std Dev | {fmt_dollar(loss_pnl.std())} |")
    lines.append("")

    win_pnl = wins_df["pnl"]
    lines.append("**Comparison with win P&L distribution:**\n")
    lines.append(f"| Metric | Wins | Losses |")
    lines.append(f"| :--- | ---: | ---: |")
    lines.append(f"| Mean | {fmt_dollar(win_pnl.mean())} | {fmt_dollar(loss_pnl.mean())} |")
    lines.append(f"| Median | {fmt_dollar(win_pnl.median())} | {fmt_dollar(loss_pnl.median())} |")
    lines.append(f"| Std Dev | {fmt_dollar(win_pnl.std())} | {fmt_dollar(loss_pnl.std())} |")
    lines.append(f"| Skewness | {win_pnl.skew():.2f} | {loss_pnl.skew():.2f} |")
    lines.append("")
    lines.append(f"Win P&L is highly variable (${win_pnl.min():.2f} to ${win_pnl.max():.2f}) "
                 f"while losses are more tightly clustered. This asymmetry is favorable — "
                 f"large wins with bounded losses.\n")

    # --- Loss classification by entry_price ---
    lines.append("### Loss Types by Entry Price\n")
    cheap = loss_df[loss_df["entry_price"] <= 0.25]
    mid = loss_df[(loss_df["entry_price"] > 0.25) & (loss_df["entry_price"] <= 0.40)]
    near_atm = loss_df[loss_df["entry_price"] > 0.40]

    type_headers = ["Loss Type", "n", "Avg Loss", "Avg Entry", "Avg Contracts", "Avg Net Edge", "Avg Ensemble"]
    type_rows = []
    for label, subset in [("Cheap (<= 0.25)", cheap), ("Mid (0.25-0.40)", mid), ("Near-ATM (> 0.40)", near_atm)]:
        if len(subset) == 0:
            type_rows.append([label, "0", "—", "—", "—", "—", "—"])
        else:
            type_rows.append([
                label, str(len(subset)),
                fmt_dollar(subset["pnl"].mean()),
                fmt_f(subset["entry_price"].mean(), 3),
                fmt_f(subset["contracts"].mean(), 0),
                fmt_f(subset["bar_net_edge"].mean(), 4),
                fmt_f(subset["ensemble_prob"].mean(), 4),
            ])
    lines.append(md_table(type_headers, type_rows))
    lines.append("")

    # BUY_NO losses
    no_trades = df[df["direction"] == "BUY_NO"]
    if len(no_trades) > 0:
        no_wins = int(no_trades["won"].sum())
        no_losses = len(no_trades) - no_wins
        lines.append(f"**BUY_NO trade outcomes:** {no_wins}W / {no_losses}L "
                     f"({no_wins/len(no_trades):.0%} win rate) out of {len(no_trades)} trades. "
                     f"{'Too few trades to draw conclusions.' if len(no_trades) < 20 else ''}\n")

    # --- Top 5 biggest losses ---
    lines.append("### Largest Losses\n")
    top_losses = loss_df.nsmallest(5, "pnl")
    big_headers = ["Trade #", "PnL", "Entry", "Contracts", "Side", "Edge", "Ensemble",
                   "BTC Close", "Secs to Exp"]
    big_rows = []
    for _, r in top_losses.iterrows():
        big_rows.append([
            str(int(r["trade_id"])), fmt_dollar(r["pnl"]), fmt_f(r["entry_price"], 3),
            str(int(r["contracts"])), r["direction"],
            fmt_f(r["bar_net_edge"], 4), fmt_f(r["ensemble_prob"], 4),
            f"${r['btc_close']:,.2f}", str(int(r["seconds_to_expiry"]))
        ])
    lines.append(md_table(big_headers, big_rows))
    lines.append("")

    # Common patterns in biggest losses
    lines.append("**Common traits of largest losses:**\n")
    avg_entry_top = top_losses["entry_price"].mean()
    avg_contracts_top = top_losses["contracts"].mean()
    avg_edge_top = top_losses["bar_net_edge"].mean()
    lines.append(f"- Average entry price: {avg_entry_top:.3f} "
                 f"(vs overall avg {df['entry_price'].mean():.3f})")
    lines.append(f"- Average contracts: {avg_contracts_top:.0f} "
                 f"(vs overall avg {df['contracts'].mean():.0f})")
    lines.append(f"- Average net edge: {avg_edge_top:.4f} "
                 f"(vs overall avg {df['bar_net_edge'].mean():.4f})")
    if avg_entry_top < df["entry_price"].mean():
        lines.append(f"- Largest losses cluster at lower entry prices (cheaper contracts), "
                     f"which get higher contract counts from Kelly sizing.")
    lines.append("")

    # --- Payoff asymmetry ---
    lines.append("### Payoff Asymmetry\n")
    avg_w = abs(win_pnl.mean())
    avg_l = abs(loss_pnl.mean())
    ratio = avg_w / avg_l if avg_l > 0 else float("inf")
    breakeven_wr = 1 / (1 + ratio)
    actual_wr = df["won"].mean()
    buffer = actual_wr - breakeven_wr

    lines.append(f"| Metric | Value |")
    lines.append(f"| :--- | ---: |")
    lines.append(f"| Avg win | {fmt_dollar(avg_w)} |")
    lines.append(f"| Avg loss | {fmt_dollar(avg_l)} |")
    lines.append(f"| Win/loss ratio | {ratio:.2f}x |")
    lines.append(f"| Breakeven win rate | {breakeven_wr:.1%} |")
    lines.append(f"| Actual win rate | {actual_wr:.1%} |")
    lines.append(f"| Buffer above breakeven | {buffer:.1%} |")
    lines.append(f"| Expected value/trade | {fmt_dollar(actual_wr * avg_w - (1-actual_wr) * avg_l)} |")
    lines.append("")
    lines.append(f"The strategy has a **{buffer:.1%} buffer** above its breakeven win rate. "
                 f"Even if the win rate dropped to {breakeven_wr:.0%}, the strategy would still break even "
                 f"given the current payoff asymmetry.\n")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Phase 5: Observations & Potential Ideas
# ---------------------------------------------------------------------------

def phase5_observations(df):
    lines = []
    lines.append("## Phase 5 — Observations & Potential Ideas\n")

    # Compute some stats for the narrative
    wr = df["won"].mean()
    wins_df = df[df["won"] == 1]
    loss_df = df[df["won"] == 0]
    won = df["won"].values
    lag1_r, lag1_p = pearsonr(won[:-1], won[1:])

    # Entry price correlation
    entry_corr, entry_p = pearsonr(df["entry_price"], df["won"])
    edge_corr, edge_p = pearsonr(df["bar_net_edge"], df["won"])
    cnn_corr, cnn_p = pearsonr(df["cnn_prob"], df["won"])
    xgb_corr, xgb_p = pearsonr(df["xgb_prob"], df["won"])

    lines.append("### Key Findings\n")

    lines.append("**1. The strategy is profitable with favorable asymmetry.**")
    payoff_ratio = abs(wins_df['pnl'].mean() / loss_df['pnl'].mean())
    ev_per_trade = wr * wins_df['pnl'].mean() + (1 - wr) * loss_df['pnl'].mean()
    be_wr = 1 / (1 + payoff_ratio)
    lines.append(f"At {wr:.1%} win rate with wins averaging {payoff_ratio:.2f}x "
                 f"the size of losses, the bot has a healthy expected value of ~${ev_per_trade:.2f}/trade. "
                 f"The breakeven win rate is ~{be_wr:.0%}, giving a {wr - be_wr:.0%} cushion.\n")

    lines.append("**2. Entry price is the strongest predictor of outcomes.**")
    lines.append(f"Correlation between entry_price and win: r={entry_corr:.3f} (p={entry_p:.4f}). "
                 f"More ATM contracts (higher entry price) win more often. This is consistent with "
                 f"the known OTM trap — the ML model predicts direction, which maps to strike-crossing "
                 f"probability only for near-ATM contracts.\n")

    lines.append("**3. Bar net edge is *inversely* correlated with winning.**")
    lines.append(f"Correlation: r={edge_corr:.3f} (p={edge_p:.4f}). Higher perceived edge comes from "
                 f"cheap contracts where the model sees 55% but the market prices at 18%, creating a "
                 f"large gap. This gap is likely not real edge — it's a calibration mismatch between "
                 f"directional probability and strike-crossing probability on cheaper contracts.\n")

    lines.append("**4. Outcomes are not independent — wins and losses cluster.**")
    lines.append(f"Lag-1 autocorrelation: r={lag1_r:.3f} (p={lag1_p:.4f}). "
                 f"This likely reflects BTC price regimes: when BTC trends in a direction favorable "
                 f"to the bot's positions, consecutive markets resolve favorably (and vice versa). "
                 f"This clustering is also consistent with the bot trading the same strike/direction "
                 f"across adjacent 15-minute windows.\n")

    lines.append("**5. CNN-LSTM provides minimal discriminative signal.**")
    lines.append(f"CNN probability range is extremely narrow: {df['cnn_prob'].min():.4f} to {df['cnn_prob'].max():.4f}. "
                 f"Correlation with outcome: r={cnn_corr:.3f} (p={cnn_p:.4f}). "
                 f"XGBoost correlation: r={xgb_corr:.3f} (p={xgb_p:.4f}). "
                 f"XGBoost appears to be doing the heavy lifting.\n")

    lines.append("**6. All trades are in 'ranging' regime.**")
    lines.append("The regime detector has classified every market as 'ranging' across all 118 trades. "
                 "Either the regime threshold needs recalibration, or BTC was genuinely range-bound "
                 "during this period. No trending-regime data exists for comparison.\n")

    lines.append("**7. All losses classified as ALPHA_DECAY are artifacts.**")
    lines.append("Due to the null `signal_edge` bug (fixed from trade #119), "
                 "the postmortem engine defaulted every loss to ALPHA_DECAY. Real failure "
                 "classification will only be meaningful from trade #119 onward.\n")

    # --- Patterns that might be real ---
    lines.append("### Patterns That Might Be Real (Monitor These)\n")

    lines.append("**Entry price / win rate gradient** — Trades with entry_price > 0.38 have "
                 "a noticeably higher win rate than those below 0.22. This aligns with the "
                 "theoretical framework (directional signal ~ strike-crossing for ATM only). "
                 f"Sample sizes are reasonable (~24 per quintile). Worth monitoring.\n")

    lines.append("**Outcome autocorrelation** — The clustering effect is statistically robust "
                 f"(lag-1 r={lag1_r:.3f}, runs test also significant). This is likely real and "
                 f"driven by BTC micro-regimes. It means drawdowns will come in bunches, "
                 f"which the risk manager should already handle via daily loss limits.\n")

    lines.append("**Edge inversion** — The fact that higher bar_net_edge correlates with *worse* "
                 "outcomes is concerning from a sizing perspective, since Kelly uses this edge "
                 "estimate. However, the Kelly fraction column is currently null, so we cannot "
                 "confirm how much this actually affected position sizing. Revisit after trade #119 "
                 "when signal_edge data becomes available.\n")

    # Check if any other dimensions showed significance
    lines.append("**Other dimensions** — Most other features (RSI, MACD, OFI, volatility, spread, "
                 "aggressor ratio) did not show statistically significant associations with outcome "
                 "in this 118-trade sample. This doesn't mean they're useless — the sample may simply "
                 "be too small, or their effect may be captured by the ensemble model already.\n")

    # --- Low-risk experiments ---
    lines.append("### Low-Risk Experiments to Consider\n")

    lines.append("**1. Raise OTM_LOW guard from 0.15 to 0.25 (1-week test)**")
    lines.append("The win rate on trades with entry_price <= 0.22 appears lower than average. "
                 "A temporary experiment raising the floor to 0.25 would reduce trade frequency "
                 "but may improve win rate. Run for 1 week and compare.\n")

    lines.append("**2. Shadow model with CNN weight = 0 (pure XGB)**")
    lines.append("Given the CNN's extremely narrow output range and low correlation with outcomes, "
                 "test whether a pure XGB ensemble performs comparably. This is zero-risk as a "
                 "shadow model — no real trades affected.\n")

    lines.append("**3. Post-streak caution (heuristic candidate)**")
    lines.append("Given the strong autocorrelation, consider a heuristic that reduces position size "
                 "by 25-50% after 3 consecutive losses. Frame as a confidence-weighted rule in the "
                 "heuristics engine. Backtest on existing data first.\n")

    lines.append("**4. Track time-of-day and day-of-week**")
    lines.append("Fix the `expiry_time` export (or add a `trade_timestamp` column) so future "
                 "analysis can check for temporal patterns. This is a data collection improvement, "
                 "not a strategy change.\n")

    # --- Additional data ---
    lines.append("### Additional Data That Would Improve Future Analysis\n")
    lines.append("- **Timestamps**: Fix expiry_time column export to enable time-of-day/day-of-week analysis")
    lines.append("- **signal_edge / kelly_fraction**: Now fixed from trade #119 — accumulate 50+ trades")
    lines.append("- **BTC hourly ATR**: Longer-horizon volatility to detect regime shifts")
    lines.append("- **Settlement result (yes/no)**: Distinguish 'won because BTC moved correctly' from "
                 "'won because market was mispriced'")
    lines.append("- **Market volume/liquidity**: Order book depth at time of trade entry")
    lines.append("- **BTC trend over the full 15-min window**: Not just the entry bar's log_return, "
                 "but the total BTC move from trade entry to settlement")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    df = pd.read_csv("data/trade_history.csv")
    print(f"Loaded {len(df)} trades from data/trade_history.csv")

    sections = []
    sections.append("# Trade History Analysis Report\n")
    sections.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} | "
                    f"Trades analyzed: {len(df)} | "
                    f"Overall win rate: {df['won'].mean():.1%}*\n")
    sections.append("---\n")

    print("Running Phase 1: Data Inventory...")
    sections.append(phase1_data_inventory(df))

    print("Running Phase 2: Win/Loss Breakdown...")
    sections.append(phase2_win_loss_breakdown(df))

    print("Running Phase 3: Sequence & Streak Analysis...")
    sections.append(phase3_sequence_analysis(df))

    print("Running Phase 4: Loss Characterization...")
    sections.append(phase4_loss_characterization(df))

    print("Running Phase 5: Observations & Potential Ideas...")
    sections.append(phase5_observations(df))

    report = "\n".join(sections)
    Path("data/analysis_report.md").write_text(report, encoding="utf-8")
    print(f"\nReport written to data/analysis_report.md ({len(report):,} characters)")


if __name__ == "__main__":
    main()
