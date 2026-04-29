import { TrendingUp, TrendingDown, Minus, AlertTriangle } from 'lucide-react'

const SIGNAL_CONFIG = {
  BUY:  { cls: 'badge-buy',  icon: <TrendingUp size={14} />,  label: 'BUY'  },
  SELL: { cls: 'badge-sell', icon: <TrendingDown size={14} />, label: 'SELL' },
  HOLD: { cls: 'badge-hold', icon: <Minus size={14} />,        label: 'HOLD' },
}

function ConfidenceArc({ confidence = 0.5, signal = 'HOLD' }) {
  const pct = Math.round(confidence * 100)
  const color = signal === 'BUY' ? 'var(--teal)' : signal === 'SELL' ? 'var(--coral)' : 'var(--amber)'
  const r = 28, circ = 2 * Math.PI * r
  const offset = circ - (pct / 100) * circ

  return (
    <div style={{ position: 'relative', width: '72px', height: '72px', flexShrink: 0 }}>
      <svg width="72" height="72" style={{ transform: 'rotate(-90deg)' }}>
        <circle cx="36" cy="36" r={r} fill="none" stroke="var(--border)" strokeWidth="5" />
        <circle cx="36" cy="36" r={r} fill="none" stroke={color} strokeWidth="5"
          strokeDasharray={circ} strokeDashoffset={offset}
          strokeLinecap="round" style={{ transition: 'stroke-dashoffset 0.6s ease' }} />
      </svg>
      <div style={{
        position: 'absolute', inset: 0, display: 'flex', flexDirection: 'column',
        alignItems: 'center', justifyContent: 'center',
      }}>
        <span className="mono" style={{ fontSize: '14px', fontWeight: 700, color }}>{pct}%</span>
      </div>
    </div>
  )
}

function Row({ label, value, valueStyle = {} }) {
  return (
    <div style={{
      display: 'flex', justifyContent: 'space-between', alignItems: 'center',
      padding: '8px 0', borderBottom: '1px solid var(--border)',
    }}>
      <span style={{ color: 'var(--text-muted)', fontSize: '12px', textTransform: 'uppercase', letterSpacing: '0.06em' }}>{label}</span>
      <span className="mono" style={{ fontSize: '13px', fontWeight: 500, ...valueStyle }}>{value}</span>
    </div>
  )
}

export default function AISignalCard({ data, loading = false }) {
  if (loading) {
    return (
      <div className="card" style={{ animationDelay: '0.1s' }}>
        <div className="skeleton" style={{ height: '200px', borderRadius: '8px' }} />
      </div>
    )
  }

  if (!data) return null

  const signal = data.trading_signals?.combined_signal || data.trading_signals?.trend_signal || 'HOLD'
  const confidence = data.trading_signals?.confidence || 0.5
  const cfg = SIGNAL_CONFIG[signal] || SIGNAL_CONFIG.HOLD

  const price = data.current_price?.toFixed(2)
  const rsi = data.technical_analysis?.rsi?.toFixed(1)
  const macd = data.technical_analysis?.macd || '—'
  const sentiment = data.sentiment_score?.toFixed(3)
  const regime = data.risk_assessment?.market_regime || data.risk_metrics?.market_regime || '—'
  const posSize = data.risk_assessment?.recommended_position_size || '—'

  const priceStr = price ? `$${price}` : '—'

  return (
    <div className="card fade-in" style={{ position: 'relative', overflow: 'hidden' }}>
      {/* Glow accent based on signal */}
      <div style={{
        position: 'absolute', top: 0, left: 0, right: 0, height: '3px',
        background: signal === 'BUY' ? 'var(--teal)' : signal === 'SELL' ? 'var(--coral)' : 'var(--amber)',
      }} />

      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '16px' }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
            <span className="ticker" style={{ color: 'var(--text-muted)' }}>{data.symbol}</span>
            <span className="ticker" style={{ color: 'var(--text-muted)', fontSize: '10px' }}>NASDAQ</span>
          </div>
          <div className="mono" style={{ fontSize: '28px', fontWeight: 700 }}>{priceStr}</div>
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '8px' }}>
          <ConfidenceArc confidence={confidence} signal={signal} />
          <span className={`badge ${cfg.cls}`} style={{ fontSize: '10px' }}>
            {cfg.icon} {cfg.label}
          </span>
        </div>
      </div>

      {/* Metrics */}
      <Row label="RSI (14)" value={rsi ? `${rsi} — ${rsi < 30 ? 'oversold' : rsi > 70 ? 'overbought' : 'neutral'}` : '—'}
        valueStyle={{ color: rsi < 30 ? 'var(--teal)' : rsi > 70 ? 'var(--coral)' : 'var(--text-primary)' }} />
      <Row label="MACD" value={typeof macd === 'string' ? macd : macd > 0 ? 'bullish' : 'bearish'}
        valueStyle={{ color: (typeof macd === 'string' ? macd.toLowerCase() : macd > 0 ? 'bullish' : 'bearish') === 'bullish' ? 'var(--teal)' : 'var(--coral)' }} />
      <Row label="News Sentiment" value={sentiment ? (sentiment > 0 ? `+${sentiment}` : sentiment) : '—'}
        valueStyle={{ color: sentiment > 0 ? 'var(--teal)' : sentiment < 0 ? 'var(--coral)' : 'var(--text-secondary)' }} />
      <Row label="Market Regime" value={regime} />
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', paddingTop: '8px' }}>
        <span style={{ color: 'var(--text-muted)', fontSize: '12px', textTransform: 'uppercase', letterSpacing: '0.06em' }}>Position Size</span>
        <span className="badge badge-buy" style={{ fontSize: '10px' }}>{posSize} of portfolio</span>
      </div>
    </div>
  )
}
