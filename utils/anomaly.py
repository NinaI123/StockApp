"""
utils/anomaly.py — Anomaly Detection Engine

Monitors three event types per ticker:

  1. volatility_spike      — current 1-day realised vol > 2σ above 30-day vol average
  2. sentiment_reversal    — sentiment compound score swings > 40 pts in 3-hour window
  3. price_lstm_divergence — actual price move deviates from LSTM predicted direction
                             by more than 2σ of the historical daily-return distribution

Each detected event is returned as a dict and optionally published to Redis.
The Redis live feed is fully optional — if Redis is unreachable the detector
still logs to the DB without error.

Redis data model
────────────────
  Key:   anomaly_feed:{TICKER}    LPUSH, LTRIM to 100 entries  (per-ticker stream)
  Key:   anomaly_feed:all         LPUSH, LTRIM to 500 entries  (global stream)
  Value: JSON-serialised anomaly record
  TTL:   24 hours per key

Severity thresholds (σ)
────────────────────────
  medium   : 2.0 – 3.0
  high     : 3.0 – 4.0
  critical : > 4.0
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import yfinance as yf

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
VOL_LOOKBACK_DAYS     = 30          # days for vol baseline
VOL_SPIKE_SIGMA       = 2.0         # z-score threshold for vol spike
SENTIMENT_SWING_PTS   = 40          # compound-score swing threshold (×100 scale)
SENTIMENT_WINDOW_MINS = 180         # 3-hour window for sentiment reversal
PRICE_DIV_SIGMA       = 2.0         # z-score threshold for price/LSTM divergence
PRICE_DIV_LOOKBACK    = 252         # trading days for return std calibration

REDIS_PER_TICKER_CAP  = 100
REDIS_GLOBAL_CAP      = 500
REDIS_TTL_SECONDS     = 86_400      # 24 h


# ── Severity helper ───────────────────────────────────────────────────────────
def _severity(sigma: float) -> str:
    if sigma >= 4.0:
        return "critical"
    if sigma >= 3.0:
        return "high"
    return "medium"


# ── Redis (optional) ──────────────────────────────────────────────────────────
_REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
_redis: Optional[object] = None          # lazy-init


def _get_redis():
    """Return a Redis client, or None if unavailable."""
    global _redis
    if _redis is not None:
        return _redis
    try:
        import redis  # type: ignore
        client = redis.from_url(_REDIS_URL, decode_responses=True, socket_connect_timeout=2)
        client.ping()
        _redis = client
        logger.info("anomaly: Redis connected at %s", _REDIS_URL)
    except Exception as exc:
        logger.warning("anomaly: Redis unavailable (%s) — live feed disabled", exc)
        _redis = False          # sentinel: don't retry every call
    return _redis if _redis else None


def _publish(record: dict) -> None:
    """Push the anomaly record to Redis (fire-and-forget)."""
    r = _get_redis()
    if not r:
        return
    try:
        payload = json.dumps(record, default=str)
        ticker  = record["ticker"]
        per_key = f"anomaly_feed:{ticker}"
        all_key = "anomaly_feed:all"
        pipe = r.pipeline()
        pipe.lpush(per_key, payload)
        pipe.ltrim(per_key, 0, REDIS_PER_TICKER_CAP - 1)
        pipe.expire(per_key, REDIS_TTL_SECONDS)
        pipe.lpush(all_key, payload)
        pipe.ltrim(all_key, 0, REDIS_GLOBAL_CAP - 1)
        pipe.expire(all_key, REDIS_TTL_SECONDS)
        pipe.execute()
    except Exception as exc:
        logger.warning("anomaly: Redis publish failed: %s", exc)


def get_live_feed(ticker: Optional[str] = None, limit: int = 50) -> list[dict]:
    """
    Read anomaly records from the Redis live feed.
    Falls back to an empty list if Redis is unavailable.
    """
    r = _get_redis()
    if not r:
        return []
    try:
        key  = f"anomaly_feed:{ticker.upper()}" if ticker else "anomaly_feed:all"
        raw  = r.lrange(key, 0, limit - 1)
        return [json.loads(x) for x in raw]
    except Exception as exc:
        logger.warning("anomaly: Redis read failed: %s", exc)
        return []


# ── Price data helper ─────────────────────────────────────────────────────────
def _fetch_closes(ticker: str, days: int) -> "pd.Series":
    import pandas as pd
    from datetime import date
    end   = date.today()
    start = end - timedelta(days=days + 10)
    raw   = yf.download(ticker, start=start.isoformat(), end=end.isoformat(),
                        progress=False, auto_adjust=True)
    if raw.empty:
        raise ValueError(f"No data for {ticker}")
    closes = raw["Close"]
    if hasattr(closes, "iloc") and closes.ndim > 1:
        closes = closes.iloc[:, 0]
    return closes.dropna()


# ═══════════════════════════════════════════════════════════════════════════════
# DETECTOR 1 — Volatility Spike
# ═══════════════════════════════════════════════════════════════════════════════

def detect_volatility_spike(ticker: str) -> Optional[dict]:
    """
    Detect if the current 1-day realised vol is more than VOL_SPIKE_SIGMA
    standard deviations above its 30-day rolling average.

    Returns an anomaly record dict, or None if no spike is detected.
    """
    import numpy as np

    # Need VOL_LOOKBACK_DAYS + buffer of history
    closes = _fetch_closes(ticker, days=VOL_LOOKBACK_DAYS + 10)
    log_ret = np.log(closes / closes.shift(1)).dropna()

    if len(log_ret) < VOL_LOOKBACK_DAYS + 1:
        return None

    # 30-day rolling vol (annualised)
    rolling_vol = log_ret.rolling(VOL_LOOKBACK_DAYS).std() * np.sqrt(252)
    rolling_vol = rolling_vol.dropna()

    if len(rolling_vol) < 2:
        return None

    current_vol   = float(rolling_vol.iloc[-1])
    baseline_mean = float(rolling_vol.iloc[:-1].mean())
    baseline_std  = float(rolling_vol.iloc[:-1].std(ddof=1))

    if baseline_std == 0:
        return None

    z = (current_vol - baseline_mean) / baseline_std

    if z < VOL_SPIKE_SIGMA:
        return None

    record = {
        "ticker":         ticker,
        "event_type":     "volatility_spike",
        "magnitude_sigma": round(z, 4),
        "severity":       _severity(z),
        "detected_at":    datetime.utcnow().isoformat(),
        "details": {
            "current_vol_annualised": round(current_vol, 6),
            "baseline_mean_vol":      round(baseline_mean, 6),
            "baseline_std_vol":       round(baseline_std, 6),
            "threshold_sigma":        VOL_SPIKE_SIGMA,
        },
    }
    _publish(record)
    return record


# ═══════════════════════════════════════════════════════════════════════════════
# DETECTOR 2 — Sentiment Reversal
# ═══════════════════════════════════════════════════════════════════════════════

def detect_sentiment_reversal(
    ticker: str,
    current_score: float,
    redis_client=None,
) -> Optional[dict]:
    """
    Detect if the sentiment compound score has swung more than SENTIMENT_SWING_PTS
    (on a ±100 scale) within the last SENTIMENT_WINDOW_MINS minutes.

    Strategy:
      - Push the current score (with timestamp) into a Redis list for this ticker.
      - Scan the list for the oldest score still within the 3-hour window.
      - If max(score) - min(score) > threshold → reversal detected.

    Falls back gracefully when Redis is unavailable (no detection, no crash).

    Parameters
    ----------
    ticker        : uppercase ticker symbol
    current_score : VADER compound score in [-1, 1]; scaled ×100 internally
    redis_client  : optional pre-existing Redis client (avoids re-connecting)
    """
    # Scale to ±100 for intuitive comparison
    scaled_current = current_score * 100.0

    r = redis_client or _get_redis()
    if not r:
        # Can't detect without Redis; return None silently
        return None

    try:
        key       = f"sentiment_history:{ticker}"
        now       = datetime.utcnow()
        cutoff    = (now - timedelta(minutes=SENTIMENT_WINDOW_MINS)).isoformat()
        entry     = json.dumps({"ts": now.isoformat(), "score": scaled_current})

        # Push current reading and trim to last 200 entries
        r.lpush(key, entry)
        r.ltrim(key, 0, 199)
        r.expire(key, REDIS_TTL_SECONDS)

        # Read back and filter to the 3-hour window
        raw_entries = r.lrange(key, 0, -1)
        window_scores = []
        for raw in raw_entries:
            obj = json.loads(raw)
            if obj["ts"] >= cutoff:
                window_scores.append(obj["score"])

        if len(window_scores) < 2:
            return None

        swing = max(window_scores) - min(window_scores)

        if swing < SENTIMENT_SWING_PTS:
            return None

        # Express magnitude as σ (normalise swing against threshold)
        # Use a conservative 1σ ≈ threshold/2 so 40-pt swing = 2σ, 80-pt = 4σ
        sigma = swing / (SENTIMENT_SWING_PTS / VOL_SPIKE_SIGMA)

        record = {
            "ticker":          ticker,
            "event_type":      "sentiment_reversal",
            "magnitude_sigma": round(sigma, 4),
            "severity":        _severity(sigma),
            "detected_at":     now.isoformat(),
            "details": {
                "swing_pts":          round(swing, 2),
                "window_minutes":     SENTIMENT_WINDOW_MINS,
                "threshold_pts":      SENTIMENT_SWING_PTS,
                "readings_in_window": len(window_scores),
                "window_max":         round(max(window_scores), 2),
                "window_min":         round(min(window_scores), 2),
            },
        }
        _publish(record)
        return record

    except Exception as exc:
        logger.warning("sentiment_reversal detection failed for %s: %s", ticker, exc)
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# DETECTOR 3 — Price / LSTM Divergence
# ═══════════════════════════════════════════════════════════════════════════════

def detect_price_lstm_divergence(
    ticker:           str,
    lstm_trend:       str,    # "UP" / "SIDEWAYS" / "DOWN"
    lstm_confidence:  float,  # 0–1 probability of the predicted class
) -> Optional[dict]:
    """
    Detect when the actual recent price move contradicts the LSTM trend forecast
    by more than PRICE_DIV_SIGMA standard deviations.

    Method
    ------
    1. Compute the 1-day log return for the ticker.
    2. Pull 252 days of daily returns to build the historical return distribution.
    3. Z-score the 1-day return: z = (ret_1d - μ) / σ_hist
    4. Map the LSTM prediction to an expected direction: +1 / 0 / -1
    5. Divergence is high when the z-score contradicts the expected direction:
         - LSTM says UP (dir = +1) but z ≪ 0  → magnitude = |z| + confidence bonus
         - LSTM says DOWN (dir = -1) but z ≫ 0 → same
         - LSTM says SIDEWAYS and |z| > threshold → flag regardless of sign
    6. If effective magnitude > PRICE_DIV_SIGMA → anomaly.
    """
    closes = _fetch_closes(ticker, days=PRICE_DIV_LOOKBACK + 10)
    log_ret = np.log(closes / closes.shift(1)).dropna()

    if len(log_ret) < 30:
        return None

    hist_returns  = log_ret.iloc[:-1]        # exclude today
    ret_today     = float(log_ret.iloc[-1])
    mu            = float(hist_returns.mean())
    sigma_hist    = float(hist_returns.std(ddof=1))

    if sigma_hist == 0:
        return None

    z_today = (ret_today - mu) / sigma_hist

    # Map LSTM direction
    trend_upper = lstm_trend.upper().strip()
    if trend_upper in ("UP", "STRONG_UP"):
        expected_dir = 1
    elif trend_upper in ("DOWN", "STRONG_DOWN"):
        expected_dir = -1
    else:
        expected_dir = 0          # SIDEWAYS

    # Compute effective divergence magnitude
    if expected_dir == 0:
        # Sideways predicted but large move → divergence = |z|
        effective_sigma = abs(z_today)
    else:
        # Directional disagreement: positive when z contradicts expected direction
        # If expected UP (+1) and z is negative → contradiction magnitude = |z|
        # Confidence scales the effective threshold downward (high-confidence wrong = worse)
        contradiction = -expected_dir * z_today          # positive when contradicting
        effective_sigma = contradiction + (lstm_confidence - 0.5) * 2.0

    if effective_sigma < PRICE_DIV_SIGMA:
        return None

    record = {
        "ticker":          ticker,
        "event_type":      "price_lstm_divergence",
        "magnitude_sigma": round(effective_sigma, 4),
        "severity":        _severity(effective_sigma),
        "detected_at":     datetime.utcnow().isoformat(),
        "details": {
            "lstm_trend":         lstm_trend,
            "lstm_confidence":    round(lstm_confidence, 4),
            "expected_direction": expected_dir,
            "return_1d_pct":      round(ret_today * 100, 4),
            "z_score_1d":         round(z_today, 4),
            "hist_return_mu":     round(mu * 100, 6),
            "hist_return_sigma":  round(sigma_hist * 100, 6),
            "threshold_sigma":    PRICE_DIV_SIGMA,
        },
    }
    _publish(record)
    return record
