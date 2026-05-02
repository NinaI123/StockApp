"""
utils/divergence.py — Signal Divergence Engine (Continuous Version)

After the LSTM/XGBoost pipeline runs, this engine:
  1. Receives raw float signals: trend_confidence, sentiment_score, technicals
  2. Maps each into a continuous range [-1.0, +1.0]
  3. Computes population variance across the three floats
  4. Scales variance to a 0–100 composite score
  5. Classifies: low (0–33) / medium (34–66) / high (67–100)

The maximum population variance of three values in [-1, 1] is 8/9 (achieved by e.g. [-1, 1, 1]).
We use this maximum to scale the score cleanly to [0, 100].
"""

from __future__ import annotations
from typing import Optional
import numpy as np

# ── Severity thresholds ───────────────────────────────────────────────────────
LOW_MAX    = 33
MEDIUM_MAX = 66

class Severity:
    LOW    = "low"
    MEDIUM = "medium"
    HIGH   = "high"

# ── Continuous Signal Mapping ────────────────────────────────────────────────

def _trend_float(trend_signal: str, confidence: float) -> float:
    """
    Map LSTM trend output to a continuous direction [-1.0, 1.0].
    """
    s = trend_signal.upper().strip()
    if s in ("UP", "STRONG_UP"):
        return float(confidence)
    if s in ("DOWN", "STRONG_DOWN"):
        return -float(confidence)
    return 0.0  # SIDEWAYS


def _technical_float(rsi: float, macd_status: str, price: float, support: float, resistance: float) -> float:
    """
    Derive a continuous technical direction [-1.0, 1.0] from RSI, MACD, and S/R bounds.
    """
    # 1. RSI: Scale 0-100 to -1 to 1. (RSI 50 = 0.0)
    rsi_norm = (rsi - 50.0) / 50.0

    # 2. MACD: Binary bullish (+1.0) / bearish (-1.0)
    macd_norm = 1.0 if macd_status.lower() == "bullish" else -1.0

    # 3. Support/Resistance Channel Position
    # Scale price between support (-1.0) and resistance (+1.0)
    if resistance > support:
        # Pct position (0.0 to 1.0)
        pct = (price - support) / (resistance - support)
        # Map to [-1, 1]
        sr_norm = (pct * 2.0) - 1.0
        # Clamp just in case price broke out of bounds
        sr_norm = max(-1.0, min(1.0, sr_norm))
    else:
        sr_norm = 0.0

    # Composite Technical Float (Weighted Average)
    # We'll weight RSI and MACD heavily, and SR channel as a modifier.
    comp = (rsi_norm * 0.4) + (macd_norm * 0.4) + (sr_norm * 0.2)
    return float(max(-1.0, min(1.0, comp)))


# ── Core computation ─────────────────────────────────────────────────────────

# Maximum possible population variance of three values bounded in [-1, +1]
# Achieved by [-1, 1, 1] or [1, -1, -1] -> mean = +/- 1/3 -> var = 8/9
_MAX_VAR: float = 8.0 / 9.0

def compute(
    trend_signal:     str,
    confidence:       float,
    sentiment_score:  float,
    rsi:              float,
    macd_status:      str,
    price:            float,
    support:          float,
    resistance:       float,
    regime:           Optional[str] = None,
) -> dict:
    """
    Compute continuous divergence score.
    """
    d_trend = _trend_float(trend_signal, confidence)
    d_sent  = float(max(-1.0, min(1.0, sentiment_score)))
    d_tech  = _technical_float(rsi, macd_status, price, support, resistance)

    directions = np.array([d_trend, d_sent, d_tech], dtype=float)

    # Population variance (ddof=0)
    variance = float(np.var(directions))

    # Scale to [0, 100]
    raw_score = (variance / _MAX_VAR) * 100.0
    score     = float(min(100.0, max(0.0, raw_score)))

    # Pairwise absolute deltas
    delta_ts = abs(d_trend - d_sent)
    delta_tt = abs(d_trend - d_tech)
    delta_st = abs(d_sent  - d_tech)

    # Severity
    if score <= LOW_MAX:
        severity = Severity.LOW
    elif score <= MEDIUM_MAX:
        severity = Severity.MEDIUM
    else:
        severity = Severity.HIGH

    return {
        # Per-signal directions (Floats)
        "signal_trend":     round(d_trend, 4),
        "signal_sentiment": round(d_sent, 4),
        "signal_technical": round(d_tech, 4),
        # Pairwise deltas
        "delta_ts": round(delta_ts, 4),
        "delta_tt": round(delta_tt, 4),
        "delta_st": round(delta_st, 4),
        # Composite score
        "variance": round(variance, 6),
        "score":    round(score,    4),
        "severity": severity,
        # Context
        "regime": regime,
        # Raw inputs (for audit / display)
        "inputs": {
            "trend_signal":     trend_signal,
            "confidence":       confidence,
            "sentiment_score":  sentiment_score,
            "rsi":              round(rsi, 4),
            "macd_status":      macd_status,
            "price":            price,
            "support":          support,
            "resistance":       resistance
        },
    }
