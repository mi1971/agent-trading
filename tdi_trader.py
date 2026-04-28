#!/usr/bin/env python3
"""
TDI Auto Trader — Traders Dynamic Index Strategy
Uses Alpaca paper trading | 4-hour bars | Fixed dollar sizing

Config file: ~/.claude/alpaca-config.json
  {
    "api_key": "YOUR_KEY",
    "secret_key": "YOUR_SECRET",
    "paper": true,
    "watchlist": ["SPY", "QQQ", "AAPL"],
    "trade_amount": 500
  }

Run:  python tdi_trader.py              (runs once immediately, then on schedule)
      python tdi_trader.py --now        (single run only, no scheduler)
"""

import os
import sys
import json
import logging
import argparse
import time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from apscheduler.schedulers.blocking import BlockingScheduler
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

try:
    import yfinance as yf
    _YFINANCE_AVAILABLE = True
except ImportError:
    _YFINANCE_AVAILABLE = False

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────

# Config: environment variables take priority (for cloud/CI runs),
# falling back to config.json for local runs.
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")

def load_config():
    # Try environment variables first (used by Claude Code Routines)
    if os.environ.get("ALPACA_API_KEY") and os.environ.get("ALPACA_SECRET_KEY"):
        return {
            "api_key":      os.environ["ALPACA_API_KEY"],
            "secret_key":   os.environ["ALPACA_SECRET_KEY"],
            "paper":        os.environ.get("ALPACA_PAPER", "true").lower() == "true",
            "watchlist":    os.environ.get("WATCHLIST", "PLTR,SE,NVDA,ARM,TSLA,AAPL,PYPL,MSTR,NFLX,ORCL").split(","),
            "trade_amount": float(os.environ.get("TRADE_AMOUNT", "1000")),
        }
    # Fall back to config.json for local runs
    with open(CONFIG_PATH) as f:
        cfg = json.load(f)
    required = ["api_key", "secret_key"]
    for key in required:
        if key not in cfg:
            raise ValueError(f"Missing '{key}' in {CONFIG_PATH}")
    if "PASTE" in cfg.get("api_key", ""):
        raise ValueError("Please add your Alpaca API key to config.json")
    return cfg

# ─────────────────────────────────────────────
#  TDI INDICATOR PARAMETERS (matches Pine Script)
# ─────────────────────────────────────────────

RSI_PERIOD    = 21      # RSI lookback
BAND_LENGTH   = 34      # Bollinger Band length on RSI
FAST_MA_BARS  = 7       # Red line — 7-bar SMA of RSI
SLOW_MA_BARS  = 2       # Green line — 2-bar SMA of RSI (more reactive)
GOLDEN_RATIO  = 1.6185  # Band multiplier (Fibonacci)

# Number of 4H bars to fetch (34 band + 21 RSI + buffer)
BARS_NEEDED   = 150

# ─────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            os.path.join(os.path.dirname(__file__), "tdi_trader.log")
        ),
    ],
)
log = logging.getLogger("TDI")

# ─────────────────────────────────────────────
#  INDICATOR CALCULATIONS
# ─────────────────────────────────────────────

def rsi(series: pd.Series, period: int) -> pd.Series:
    """Wilder RSI (matches TradingView's rsi() function)."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    # Use EWM with Wilder smoothing (com = period - 1)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calculate_tdi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all TDI components on a bar dataframe.
    Returns dataframe with added columns:
      rsi, upper, lower, mid, fast_ma, slow_ma,
      buy_signal, sell_signal
    """
    df = df.copy()

    # Core RSI
    df["rsi"] = rsi(df["close"], RSI_PERIOD)

    # Bollinger Bands around RSI
    df["bb_ma"]  = df["rsi"].rolling(BAND_LENGTH).mean()
    df["bb_std"] = df["rsi"].rolling(BAND_LENGTH).std(ddof=0)
    df["upper"]  = df["bb_ma"] + GOLDEN_RATIO * df["bb_std"]
    df["lower"]  = df["bb_ma"] - GOLDEN_RATIO * df["bb_std"]
    df["mid"]    = (df["upper"] + df["lower"]) / 2

    # Signal MAs
    df["fast_ma"] = df["rsi"].rolling(FAST_MA_BARS).mean()  # 7-bar (red)
    df["slow_ma"] = df["rsi"].rolling(SLOW_MA_BARS).mean()  # 2-bar (green)

    # Previous bar values for crossover detection
    df["prev_slow"] = df["slow_ma"].shift(1)
    df["prev_fast"] = df["fast_ma"].shift(1)

    # ── BUY: green (slow_ma) crosses ABOVE red (fast_ma)
    #         AND RSI is above the midline (bullish bias)
    #         AND slow_ma itself is above the midline
    df["buy_signal"] = (
        (df["slow_ma"] > df["fast_ma"]) &
        (df["prev_slow"] <= df["prev_fast"]) &
        (df["rsi"] > 50) &
        (df["slow_ma"] > df["mid"])
    )

    # ── SELL: green (slow_ma) crosses BELOW red (fast_ma)
    #          AND RSI is below the midline (bearish bias)
    #          AND slow_ma itself is below the midline
    df["sell_signal"] = (
        (df["slow_ma"] < df["fast_ma"]) &
        (df["prev_slow"] >= df["prev_fast"]) &
        (df["rsi"] < 50) &
        (df["slow_ma"] < df["mid"])
    )

    return df

# ─────────────────────────────────────────────
#  ALPACA HELPERS
# ─────────────────────────────────────────────

def get_clients(cfg):
    paper = cfg.get("paper", True)
    trading = TradingClient(
        api_key=cfg["api_key"],
        secret_key=cfg["secret_key"],
        paper=paper,
    )
    data = StockHistoricalDataClient(
        api_key=cfg["api_key"],
        secret_key=cfg["secret_key"],
    )
    return trading, data


def fetch_bars(data_client, symbol: str, bars: int = BARS_NEEDED) -> pd.DataFrame:
    """Fetch 1H bars via Alpaca/IEX and resample to 4H.
    Falls back to yfinance if the Alpaca data API is unavailable.
    Returns an empty DataFrame if both sources fail."""
    try:
        return _fetch_bars_alpaca(data_client, symbol, bars)
    except Exception as e:
        log.warning(f"  Alpaca data unavailable for {symbol} ({e}) — trying yfinance fallback")

    try:
        return _fetch_bars_yfinance(symbol, bars)
    except Exception as e:
        log.warning(f"  yfinance also unavailable for {symbol} ({e}) — skipping")
        return pd.DataFrame()


def _fetch_bars_alpaca(data_client, symbol: str, bars: int) -> pd.DataFrame:
    """Fetch via Alpaca IEX feed (primary path)."""
    end   = datetime.now(timezone.utc)
    start = end - timedelta(days=int(bars * 4 / 6.5) + 45)

    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Hour,
        start=start,
        end=end,
        feed="iex",
    )
    barset = data_client.get_stock_bars(req)
    df = barset.df

    if df.empty:
        return df

    if isinstance(df.index, pd.MultiIndex):
        df = df.xs(symbol, level="symbol")

    df = df[["open", "high", "low", "close", "volume"]].copy()
    df.index = pd.to_datetime(df.index, utc=True)
    df.sort_index(inplace=True)

    df_4h = df.resample("4h").agg({
        "open":   "first",
        "high":   "max",
        "low":    "min",
        "close":  "last",
        "volume": "sum",
    }).dropna(subset=["close"])

    return df_4h.tail(bars)


def _fetch_bars_yfinance(symbol: str, bars: int) -> pd.DataFrame:
    """Fetch via yfinance (fallback when Alpaca data is unavailable)."""
    if not _YFINANCE_AVAILABLE:
        raise RuntimeError("yfinance not installed; run: pip install yfinance")

    tk = yf.Ticker(symbol)
    df = tk.history(period="60d", interval="1h", auto_adjust=True)
    if df.empty:
        return df

    df.columns = [c.lower() for c in df.columns]
    df.index = pd.to_datetime(df.index, utc=True)
    df.sort_index(inplace=True)

    df_4h = df[["open", "high", "low", "close", "volume"]].resample("4h").agg({
        "open":   "first",
        "high":   "max",
        "low":    "min",
        "close":  "last",
        "volume": "sum",
    }).dropna(subset=["close"])

    return df_4h.tail(bars)


def get_position(trading_client, symbol: str):
    """Return current position for symbol, or None."""
    try:
        return trading_client.get_open_position(symbol)
    except Exception:
        return None


def market_open(trading_client) -> bool:
    """Check if the US market is currently open.
    Returns True (assume open) if the trading API is unreachable and --now is set."""
    try:
        clock = trading_client.get_clock()
        return clock.is_open
    except Exception as e:
        if getattr(run_strategy, "_force", False):
            log.warning(f"Could not reach Alpaca clock API ({e}) — assuming market open (--now override)")
            return True
        raise


def place_order(trading_client, symbol: str, side: OrderSide, dollar_amount: float):
    """Submit a notional (dollar-based) market order."""
    req = MarketOrderRequest(
        symbol=symbol,
        notional=round(dollar_amount, 2),
        side=side,
        time_in_force=TimeInForce.DAY,
    )
    order = trading_client.submit_order(req)
    log.info(
        f"  ✅ ORDER PLACED: {side.value.upper()} ${dollar_amount:,.0f} of {symbol} "
        f"(order_id={order.id})"
    )
    return order

# ─────────────────────────────────────────────
#  CORE TRADING LOGIC
# ─────────────────────────────────────────────

def run_strategy(cfg):
    """Run one full cycle of the TDI strategy across the watchlist."""
    log.info("=" * 60)
    log.info(f"TDI Strategy cycle — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info("=" * 60)

    watchlist    = cfg.get("watchlist", ["SPY"])
    trade_amount = float(cfg.get("trade_amount", 500))
    paper        = cfg.get("paper", True)

    trading_client, data_client = get_clients(cfg)

    # Check market hours (bypass with --now flag for testing)
    if not market_open(trading_client) and not getattr(run_strategy, "_force", False):
        log.info("Market is closed — skipping this cycle.")
        return

    # Attempt to load account info; degrade gracefully if trading API is blocked
    account = None
    trading_api_ok = True
    try:
        account = trading_client.get_account()
        log.info(
            f"Account | Equity: ${float(account.equity):,.2f} | "
            f"Buying Power: ${float(account.buying_power):,.2f} | "
            f"Mode: {'PAPER' if paper else 'LIVE'}"
        )
    except Exception as e:
        trading_api_ok = False
        log.warning(
            f"Trading API unavailable ({e}) — signals will be computed but orders SKIPPED."
        )

    # ── Summary table header ──────────────────────────────────────
    log.info("")
    log.info(f"{'Ticker':<6} {'Price':>8} {'RSI':>6} {'FastMA':>7} {'SlowMA':>7} {'Mid':>6}  Signal    Order")
    log.info("─" * 72)

    for symbol in watchlist:

        # 1. Fetch bars
        df = fetch_bars(data_client, symbol)
        if df.empty or len(df) < BAND_LENGTH + RSI_PERIOD:
            log.warning(f"  {symbol:<6} — not enough bars, skipping.")
            continue

        # 2. Calculate TDI
        df = calculate_tdi(df)
        latest = df.iloc[-1]

        buy_sig  = bool(latest["buy_signal"])
        sell_sig = bool(latest["sell_signal"])
        signal   = "BUY" if buy_sig else ("SELL" if sell_sig else "HOLD")

        # 3. Check current position (best-effort)
        position = get_position(trading_client, symbol) if trading_api_ok else None
        qty      = float(position.qty) if position else 0.0
        has_long = qty > 0

        # 4. Determine order action
        order_note = "—"
        if not trading_api_ok:
            order_note = f"WOULD {signal} ${trade_amount:,.0f}" if signal != "HOLD" else "—"
        elif qty < 0:
            order_note = "SHORT — skip (close manually)"
        elif buy_sig and not has_long:
            if float(account.buying_power) >= trade_amount:
                place_order(trading_client, symbol, OrderSide.BUY, trade_amount)
                order_note = f"BUY ${trade_amount:,.0f} placed"
            else:
                order_note = f"BUY skipped (low BP: ${float(account.buying_power):,.0f})"
        elif sell_sig and has_long:
            pos_value     = float(position.market_value)
            sell_notional = min(trade_amount, pos_value)
            place_order(trading_client, symbol, OrderSide.SELL, sell_notional)
            order_note = f"SELL ${sell_notional:,.0f} placed"
        elif buy_sig and has_long:
            order_note = "already long — hold"
        elif sell_sig and not has_long:
            order_note = "no position — skip"

        log.info(
            f"{symbol:<6} {latest['close']:>8.2f} {latest['rsi']:>6.1f} "
            f"{latest['fast_ma']:>7.1f} {latest['slow_ma']:>7.1f} {latest['mid']:>6.1f}  "
            f"{signal:<8}  {order_note}"
        )

        time.sleep(0.3)  # gentle rate limiting between symbols

    log.info(f"\nCycle complete.\n")

# ─────────────────────────────────────────────
#  ENTRYPOINT
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="TDI Auto Trader")
    parser.add_argument("--now", action="store_true", help="Run once and exit")
    args = parser.parse_args()

    cfg = load_config()
    paper = cfg.get("paper", True)
    watchlist = cfg.get("watchlist", [])

    log.info("TDI Auto Trader starting up")
    log.info(f"Mode:      {'PAPER' if paper else '⚠️  LIVE'}")
    log.info(f"Watchlist: {watchlist}")
    log.info(f"Trade amt: ${cfg.get('trade_amount', 500):,} per signal")
    log.info(f"Timeframe: 4-hour bars")
    log.info(f"Config:    {CONFIG_PATH}")

    if args.now:
        run_strategy._force = True   # bypass market-hours check for test runs
        run_strategy(cfg)
        return

    # Run once immediately, then every 4 hours aligned to market hours
    run_strategy(cfg)

    scheduler = BlockingScheduler(timezone=ZoneInfo("America/New_York"))
    # 4-hour bars: fire at 9:30, 13:30, 17:30 ET (covers market open & midday)
    scheduler.add_job(
        lambda: run_strategy(load_config()),  # reload config each run
        "cron",
        day_of_week="mon-fri",
        hour="9,13,17",
        minute="30",
    )

    log.info("Scheduler started — firing at 9:30, 13:30, 17:30 ET on weekdays.")
    log.info("Press Ctrl+C to stop.\n")

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        log.info("Shutting down.")


if __name__ == "__main__":
    main()
