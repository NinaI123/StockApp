import sys
import os
import yfinance as yf
import pandas as pd

# Fix path
sys.path.append(os.getcwd())

from utils.feature_engineering import FeatureEngineer

def test_fe():
    print("Fetching data...")
    df = yf.download("AAPL", period="1y", interval="1d", progress=False)
    
    # Fix MultiIndex if present (yfinance behavior)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    print(f"Data shape: {df.shape}")
    
    print("Processing features...")
    df_processed = FeatureEngineer.process_all(df)
    
    print(f"Processed shape: {df_processed.shape}")
    print("New Columns:")
    print(df_processed.columns.tolist())
    
    # Check specific columns exist
    required = ['RSI', 'MACD', 'BB_Upper', 'ATR', 'OBV', 'VIX', 'DayOfWeek']
    missing = [col for col in required if col not in df_processed.columns]
    
    if missing:
        print(f"FAILED: Missing columns: {missing}")
    else:
        print("SUCCESS: All required columns present.")
        print("Sample Data:")
        print(df_processed[['Close', 'RSI', 'MACD', 'VIX']].tail())

if __name__ == "__main__":
    test_fe()
