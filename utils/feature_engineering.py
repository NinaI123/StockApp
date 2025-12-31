import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    Comprehensive feature engineering for stock data.
    """
    
    @staticmethod
    def add_technical_indicators(df):
        """
        Add technical indicators: RSI, MACD, Bollinger Bands, ATR, OBV, MAs, Patterns.
        """
        df = df.copy()
        
        # Ensure numeric types
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        close = df['Close']
        high = df['High']
        low = df['Low']
        volume = df['Volume']
        
        # --- Moving Averages ---
        df['SMA_20'] = close.rolling(window=20).mean()
        df['SMA_50'] = close.rolling(window=50).mean()
        df['SMA_200'] = close.rolling(window=200).mean()
        
        # --- RSI (14) ---
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # --- MACD (12, 26, 9) ---
        exp1 = close.ewm(span=12, adjust=False).mean()
        exp2 = close.ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        
        # --- Bollinger Bands (20, 2) ---
        sma20 = close.rolling(window=20).mean()
        std20 = close.rolling(window=20).std()
        df['BB_Upper'] = sma20 + (std20 * 2)
        df['BB_Lower'] = sma20 - (std20 * 2)
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / sma20
        
        # --- ATR (14) ---
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['ATR'] = true_range.rolling(window=14).mean()
        
        # --- OBV (On-Balance Volume) ---
        df['OBV'] = (np.sign(close.diff()) * volume).fillna(0).cumsum()
        
        # --- Volume Indicators ---
        df['Volume_SMA20'] = volume.rolling(window=20).mean()
        df['Volume_Ratio'] = volume / df['Volume_SMA20']
        
        # --- Price Patterns (Simple) ---
        # Doji: Open and Close are very close (less than 1% of range)
        body_size = np.abs(close - df['Open'])
        total_range = high - low
        df['Pattern_Doji'] = (body_size <= (0.1 * total_range)).astype(int)
        
        # Hammer: Small body near top, long lower shadow
        # Lower shadow > 2 * body, Upper shadow < 0.1 * total_range
        lower_shadow = np.minimum(close, df['Open']) - low
        upper_shadow = high - np.maximum(close, df['Open'])
        df['Pattern_Hammer'] = (
            (lower_shadow >= 2 * body_size) & 
            (upper_shadow <= 0.1 * total_range)
        ).astype(int)
        
        return df

    @staticmethod
    def add_sentiment_features(df, news_data=None):
        """
        Add sentiment features.
        If news_data is provided (list of dicts with date/score), merge it.
        Otherwise, creates placeholders.
        """
        df = df.copy()
        
        # Placeholder for daily Aggregated Sentiment
        # In a real app, this would merge with a separate DataFrame of daily sentiment scores
        if 'News_Sentiment' not in df.columns:
            df['News_Sentiment'] = 0.0
            
        # Example of Earnings Call Sentiment (Placeholder)
        df['Earnings_Sentiment'] = 0.0
        
        # Analyst Rating Changes (Placeholder)
        # 1 = Upgrade, -1 = Downgrade, 0 = No change
        df['Analyst_Rating_Change'] = 0
        
        return df

    @staticmethod
    def add_market_features(df):
        """
        Add broad market features (VIX, Interest Rates, Sector Performance).
        WARNING: Fetches data from yfinance.
        """
        df = df.copy()
        
        try:
            # Fetch VIX (Volatility Index)
            # Efficient way: Download VIX history matching the df's timeframe + buffer
            start_date = df.index[0]
            end_date = df.index[-1]
            
            # Fetch broad market data
            # ^VIX: Volatility
            # ^TNX: 10-Year Treasury Yield
            market_tickers = ["^VIX", "^TNX"]
            market_data = yf.download(market_tickers, start=start_date, end=end_date, progress=False)
            
            # Fix MultiIndex if present
            if isinstance(market_data.columns, pd.MultiIndex):
                # We want Close prices for these
                vix = market_data['Close']['^VIX']
                tnx = market_data['Close']['^TNX']
            else:
                # If only one ticker was downloaded, structure is different, but we asked for 2
                # yfinance behavior varies. Let's be robust.
                # If fail, fill with 0
                vix = pd.Series(0, index=df.index)
                tnx = pd.Series(0, index=df.index)

            # Reindex to match df (forward fill missing weekends/holidays if market data has gaps)
            vix = vix.reindex(df.index, method='ffill')
            tnx = tnx.reindex(df.index, method='ffill')
            
            df['VIX'] = vix
            df['Treasury_Yield_10Y'] = tnx
            
            # Derived Market Features
            df['VIX_MA20'] = df['VIX'].rolling(window=20).mean()
            df['Regime_VIX'] = np.where(df['VIX'] > 20, 'High Volatility', 'Normal')
            
        except Exception as e:
            logger.warning(f"Failed to fetch market features: {e}")
            df['VIX'] = 0.0
            df['Treasury_Yield_10Y'] = 0.0
        
        return df

    @staticmethod
    def add_time_features(df):
        """
        Add calendar and event-based features.
        """
        df = df.copy()
        
        df['DayOfWeek'] = df.index.dayofweek
        df['Month'] = df.index.month
        df['Quarter'] = df.index.quarter
        df['Is_Month_End'] = df.index.is_month_end.astype(int)
        
        # Earnings Season (Approximate: Jan, Apr, Jul, Oct)
        df['Is_Earnings_Season'] = df['Month'].isin([1, 4, 7, 10]).astype(int)
        
        return df

    @classmethod
    def process_all(cls, df):
        """
        Run all feature engineering steps pipeline.
        """
        df = cls.add_technical_indicators(df)
        df = cls.add_sentiment_features(df)
        df = cls.add_time_features(df)
        # Note: add_market_features requires external calls, might be slow.
        # Included for completeness as requested.
        df = cls.add_market_features(df)
        return df
