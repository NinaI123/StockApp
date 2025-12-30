import pandas as pd
import numpy as np

def add_technical_indicators(df):
    """
    Add technical indicators to the dataframe.
    """
    df = df.copy()
    
    # Ensure Close is float
    close_prices = df['Close'].astype(float)
    high_prices = df['High'].astype(float)
    low_prices = df['Low'].astype(float)
    
    # 1. RSI (Relative Strength Index)
    delta = close_prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # 2. MACD (Moving Average Convergence Divergence)
    exp1 = close_prices.ewm(span=12, adjust=False).mean()
    exp2 = close_prices.ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # 3. Bollinger Bands
    sma20 = close_prices.rolling(window=20).mean()
    std20 = close_prices.rolling(window=20).std()
    df['BB_Upper'] = sma20 + (std20 * 2)
    df['BB_Lower'] = sma20 - (std20 * 2)
    
    # 4. ATR (Average True Range) - Volatility Measure
    high_low = high_prices - low_prices
    high_close = np.abs(high_prices - close_prices.shift())
    low_close = np.abs(low_prices - close_prices.shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['ATR'] = true_range.rolling(window=14).mean()
    
    # 5. Volume Trends
    df['Volume_SMA20'] = df['Volume'].rolling(window=20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA20']
    
    return df

def add_time_features(df):
    """Add time-based features"""
    df = df.copy()
    df['DayOfWeek'] = df.index.dayofweek
    df['Month'] = df.index.month
    df['Quarter'] = df.index.quarter
    return df

def add_market_regime(df):
    """
    Add volatility-based market regime.
    0: Low Volatility
    1: High Volatility
    """
    df = df.copy()
    # Simple regime based on ATR vs historical mean ATR
    if 'ATR' not in df.columns:
        df = add_technical_indicators(df)
        
    atr_mean = df['ATR'].rolling(window=100).mean()
    df['Regime_Vol'] = (df['ATR'] > atr_mean).astype(int)
    
    return df
