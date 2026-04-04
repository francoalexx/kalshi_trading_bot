"""Email notifications for trade settlements via Resend."""

import os

import resend
from loguru import logger


def send_settlement_email(trade, bankroll, start_bankroll, settled_trades):
    """
    Send an email notification when a trade settles.

    Args:
        trade: The PaperTrade that just settled.
        bankroll: Current bankroll after settlement.
        start_bankroll: Starting bankroll.
        settled_trades: List of all settled PaperTrades (for dashboard section).
    """
    api_key = os.getenv("RESEND_API_KEY", "")
    to_email = os.getenv("NOTIFICATION_EMAIL", "")
    from_email = os.getenv("NOTIFICATION_FROM", "trades@resend.dev")

    if not api_key or not to_email:
        return

    resend.api_key = api_key

    result_str = "WIN" if trade.won else "LOSS"
    subject = f"Trade #{trade.trade_id} — {result_str} {trade.pnl:+.2f}"

    html = _build_email_html(trade, bankroll, start_bankroll, settled_trades)

    try:
        resend.Emails.send({
            "from": from_email,
            "to": [to_email],
            "subject": subject,
            "html": html,
        })
        logger.info(f"Settlement email sent for trade #{trade.trade_id}")
    except Exception as e:
        logger.warning(f"Email notification failed: {e}")


def _build_email_html(trade, bankroll, start_bankroll, settled_trades):
    """Build the HTML email body with trade highlight + dashboard."""
    wins = sum(1 for t in settled_trades if t.won)
    losses = len(settled_trades) - wins
    win_rate = wins / len(settled_trades) if settled_trades else 0
    total_pnl = sum(t.pnl for t in settled_trades)
    total_fee = sum(t.fee for t in settled_trades)
    roi = (bankroll - start_bankroll) / start_bankroll * 100

    # Trade result styling
    if trade.won:
        result_label = "WIN"
        result_color = "#16a34a"
        bg_color = "#f0fdf4"
        border_color = "#bbf7d0"
    else:
        result_label = "LOSS"
        result_color = "#dc2626"
        bg_color = "#fef2f2"
        border_color = "#fecaca"

    # Last 10 settled trades table rows
    trade_rows = ""
    for t in settled_trades[-10:]:
        res = "WIN" if t.won else "LOSS"
        res_color = "#16a34a" if t.won else "#dc2626"
        trade_rows += f"""
        <tr>
            <td style="padding:4px 8px;text-align:right">{t.trade_id}</td>
            <td style="padding:4px 8px">{t.side.upper()}</td>
            <td style="padding:4px 8px;text-align:right">{t.contracts}</td>
            <td style="padding:4px 8px;text-align:right">{t.entry_price:.3f}</td>
            <td style="padding:4px 8px;text-align:right">{t.pnl:+.2f}</td>
            <td style="padding:4px 8px;color:{res_color};font-weight:bold">{res}</td>
        </tr>"""

    html = f"""
    <div style="font-family:monospace,monospace;max-width:600px;margin:0 auto;color:#1a1a1a">

        <!-- Trade Result Header -->
        <div style="background:{bg_color};border:1px solid {border_color};border-radius:8px;padding:16px;margin-bottom:16px">
            <div style="font-size:24px;font-weight:bold;color:{result_color};margin-bottom:8px">
                {result_label} &mdash; Trade #{trade.trade_id}
            </div>
            <table style="font-size:14px">
                <tr><td style="padding:2px 12px 2px 0;color:#666">Side</td><td><b>{trade.side.upper()}</b></td></tr>
                <tr><td style="padding:2px 12px 2px 0;color:#666">Contracts</td><td><b>{trade.contracts}</b></td></tr>
                <tr><td style="padding:2px 12px 2px 0;color:#666">Entry</td><td><b>{trade.entry_price:.3f}</b></td></tr>
                <tr><td style="padding:2px 12px 2px 0;color:#666">P&amp;L</td><td style="color:{result_color}"><b>${trade.pnl:+.2f}</b></td></tr>
                <tr><td style="padding:2px 12px 2px 0;color:#666">Fee</td><td>${trade.fee:.2f}</td></tr>
            </table>
        </div>

        <!-- Dashboard Summary -->
        <div style="border:1px solid #e5e7eb;border-radius:8px;padding:16px;margin-bottom:16px">
            <div style="font-size:16px;font-weight:bold;margin-bottom:8px;color:#374151">Dashboard</div>
            <table style="font-size:14px;width:100%">
                <tr><td style="padding:2px 0;color:#666">Bankroll</td><td style="text-align:right"><b>${bankroll:.2f}</b> (started ${start_bankroll:.2f})</td></tr>
                <tr><td style="padding:2px 0;color:#666">ROI</td><td style="text-align:right"><b>{roi:+.2f}%</b></td></tr>
                <tr><td style="padding:2px 0;color:#666">Net P&amp;L</td><td style="text-align:right"><b>${total_pnl:+.2f}</b> (fees: ${total_fee:.2f})</td></tr>
                <tr><td style="padding:2px 0;color:#666">Record</td><td style="text-align:right"><b>{wins}W / {losses}L</b> ({win_rate:.1%})</td></tr>
            </table>
        </div>

        <!-- Last 10 Trades -->
        <div style="border:1px solid #e5e7eb;border-radius:8px;padding:16px">
            <div style="font-size:16px;font-weight:bold;margin-bottom:8px;color:#374151">Recent Trades</div>
            <table style="font-size:13px;width:100%;border-collapse:collapse">
                <tr style="border-bottom:1px solid #e5e7eb;color:#666">
                    <th style="padding:4px 8px;text-align:right">#</th>
                    <th style="padding:4px 8px;text-align:left">Side</th>
                    <th style="padding:4px 8px;text-align:right">Qty</th>
                    <th style="padding:4px 8px;text-align:right">Entry</th>
                    <th style="padding:4px 8px;text-align:right">P&amp;L</th>
                    <th style="padding:4px 8px;text-align:left">Result</th>
                </tr>
                {trade_rows}
            </table>
        </div>

    </div>
    """
    return html
