"""
utils/regime.py — Rule-based market regime classifier.

Classification priority (first match wins):
  1. HIGH_VOLATILITY  — 20-day realized vol > 80th pct of its 1-year rolling history
  2. TRENDING_UP      — price > 50-day SMA  AND  SMA slope > 0
  3. TRENDING_DOWN    — price < 50-day SMA  AND  SMA slope < 0
  4. MEAN_REVERTING   — everything else

Returns a dict with the label plus all intermediate signals so callers can
store / surface the raw metrics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import date, timedelta


# ── Constants ─────────────────────────────────────────────────────────────────
VOL_WINDOW        = 20          # days for realized-volatility calculation
MA_WINDOW         = 50          # days for moving-average trend signal
SLOPE_WINDOW      = 5           # days used to estimate MA slope (end minus start)
VOL_HISTORY_DAYS  = 365         # 1-year lookback for percentile calibration
VOL_PCT_THRESHOLD = 80          # percentile threshold for high-vol flag
TRADING_DAYS_YEAR = 252         # annualisation factor


# ── Regimes ───────────────────────────────────────────────────────────────────
class Regime:
    HIGH_VOLATILITY  = "high_volatility"
    TRENDING_UP      = "trending_up"
    TRENDING_DOWN    = "trending_down"
    MEAN_REVERTING   = "mean_reverting"


# ── Core classifier ───────────────────────────────────────────────────────────

def _fetch_closes(ticker: str, days: int) -> pd.Series:
    """Download adjusted close prices for the last `days` calendar days."""
    end   = date.today()
    start = end - timedelta(days=days + 30)   # buffer for weekends / holidays
    raw   = yf.download(ticker, start=start.isoformat(), end=end.isoformat(),
                        progress=False, auto_adjust=True)
    if raw.empty:
        raise ValueError(f"No price data returned for {ticker!r}")
    closes = raw["Close"]
    # yfinance can return a DataFrame with MultiIndex columns for single tickers
    if isinstance(closes, pd.DataFrame):
        closes = closes.iloc[:, 0]
    closes = closes.dropna()
    return closes


def classify(ticker: str) -> dict:
    """
    Classify `ticker` into one of four regimes using the latest available data.

    Returns
    -------
    dict with keys:
        ticker        str
        as_of         str  (ISO date of the most recent bar used)
        regime        str  (one of the four Regime constants)
        vol_20d       float  (annualised 20-day realised vol, as a fraction, e.g. 0.32)
        vol_pct_rank  float  (0-100 percentile rank within the 1-year history)
        ma_50d        float  (current 50-day SMA value)
        ma_slope      float  (SMA change over last SLOPE_WINDOW days, normalised by SMA)
        price         float  (latest close)
    """
    ticker = ticker.upper().strip()

    # Need at least 1 year + MA window of history
    closes = _fetch_closes(ticker, days=VOL_HISTORY_DAYS + MA_WINDOW + 30)

    if len(closes) < MA_WINDOW + VOL_WINDOW:
        raise ValueError(
            f"Insufficient history for {ticker}: got {len(closes)} bars, "
            f"need at least {MA_WINDOW + VOL_WINDOW}."
        )

    # ── 1. 20-day realised volatility (annualised) ────────────────────────────
    log_returns   = np.log(closes / closes.shift(1)).dropna()
    vol_series    = log_returns.rolling(VOL_WINDOW).std() * np.sqrt(TRADING_DAYS_YEAR)
    vol_series    = vol_series.dropna()

    current_vol   = float(vol_series.iloc[-1])

    # Percentile rank of today's vol within the 1-year rolling history
    # Use only the last VOL_HISTORY_DAYS worth of vol readings for calibration
    cal_vols      = vol_series.iloc[-VOL_HISTORY_DAYS:]
    vol_pct_rank  = float((cal_vols < current_vol).mean() * 100)

    # ── 2. 50-day moving average + slope ─────────────────────────────────────
    ma_series     = closes.rolling(MA_WINDOW).mean().dropna()

    if len(ma_series) < SLOPE_WINDOW:
        raise ValueError(f"Not enough MA values to compute slope for {ticker}.")

    current_ma    = float(ma_series.iloc[-1])
    # Normalise slope by the MA itself → dimensionless (% per bar)
    ma_slope      = float(
        (ma_series.iloc[-1] - ma_series.iloc[-SLOPE_WINDOW]) / ma_series.iloc[-SLOPE_WINDOW]
    )

    current_price = float(closes.iloc[-1])
    as_of         = closes.index[-1].strftime("%Y-%m-%d")

    # ── 3. Classification (priority order) ───────────────────────────────────
    if vol_pct_rank > VOL_PCT_THRESHOLD:
        regime = Regime.HIGH_VOLATILITY
    elif current_price > current_ma and ma_slope > 0:
        regime = Regime.TRENDING_UP
    elif current_price < current_ma and ma_slope < 0:
        regime = Regime.TRENDING_DOWN
    else:
        regime = Regime.MEAN_REVERTING

    return {
        "ticker":       ticker,
        "as_of":        as_of,
        "regime":       regime,
        "vol_20d":      round(current_vol, 6),
        "vol_pct_rank": round(vol_pct_rank, 2),
        "ma_50d":       round(current_ma, 4),
        "ma_slope":     round(ma_slope, 6),
        "price":        round(current_price, 4),
    }
