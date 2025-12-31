import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def get_stock_summary(ticker, history_data):
    """Generate summary information for a stock"""
    if history_data.empty:
        return None
    
    last_close = history_data['Close'].iloc[-1]
    first_close = history_data['Close'].iloc[0]
    change = last_close - first_close
    change_pct = (change / first_close) * 100
    
    return {
        'symbol': ticker.ticker,
        'currentPrice': round(last_close, 2),
        'change': round(change, 2),
        'changePercent': round(change_pct, 2),
        'open': round(history_data['Open'].iloc[-1], 2),
        'high': round(history_data['High'].iloc[-1], 2),
        'low': round(history_data['Low'].iloc[-1], 2),
        'volume': int(history_data['Volume'].iloc[-1]),
        'marketCap': getattr(ticker.info, 'marketCap', None),
        'peRatio': getattr(ticker.info, 'trailingPE', None),
        'dividendYield': getattr(ticker.info, 'dividendYield', None),
        '52WeekHigh': getattr(ticker.info, 'fiftyTwoWeekHigh', None),
        '52WeekLow': getattr(ticker.info, 'fiftyTwoWeekLow', None),
        'lastUpdated': datetime.now().isoformat()
    }

def fetch_historical_data(symbol, period='1y'):
    """Fetch historical data with proper formatting. """
    stock = yf.Ticker(symbol)
    hist = stock.history(period=period)
    
    if hist.empty:
        return None
    
    # Convert to list format for Chart.js
    dates = hist.index.strftime('%Y-%m-%d').tolist()
    prices = hist['Close'].round(2).tolist()
    
    return {
        'symbol': symbol,
        'dates': dates,
        'prices': prices,
        'open': hist['Open'].round(2).tolist(),
        'high': hist['High'].round(2).tolist(),
        'low': hist['Low'].round(2).tolist(),
        'volume': hist['Volume'].tolist()
    }