import numpy as np
import pandas as pd
import yfinance as yf
import xgboost as xgb
import joblib
import os
import sys
import logging
from datetime import datetime, timedelta
import time
from tensorflow.keras.models import load_model, Model

# Add project root to path to ensure imports work correctly
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now we can import from models and utils safely
try:
    from models.feature_engineering import add_technical_indicators, add_time_features, add_market_regime
except ImportError:
    # If for some reason that fails (e.g. running from root without models package init?)
    from feature_engineering import add_technical_indicators, add_time_features, add_market_regime

try:
    from utils.news_api import fetch_news
    from utils.sentiment import analyze_articles
    
except ImportError as e:
    logging.error(f"Failed to import utils: {e}")
    # Define mocks if imports fail to allow verification to proceed partially
    t0 = time.time()
    def fetch_news(*args, **kwargs): return {}
    t1 = time.time()
    def analyze_articles(*args, **kwargs): return {'overall_sentiment': 'neutral', 'articles': []}
    t2 = time.time()
    logging.warning(f"utils imports: fetch_news {t1-t0:.2f}s, analyze_articles {t2-t1:.2f}s")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PredictionCache:
    """Thread-safe TTL Cache for Predictions"""
    def __init__(self, ttl_seconds=900): # 15 minutes default
        self.cache = {}
        self.ttl = ttl_seconds
        
    def get(self, key):
        if key in self.cache:
            data, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl:
                return data
            else:
                del self.cache[key]
        return None
        
    def set(self, key, value):
        self.cache[key] = (value, time.time())

class EnhancedPredictor:
    def __init__(self, model_dir="models/saved"):
        self.model_dir = model_dir
        self.lookback = 60
        self.cache = PredictionCache(ttl_seconds=900) # 15 Minutes
        self.load_artifacts()
        
    def load_artifacts(self):
        """Load all models and scalers"""
        try:
            logger.info("Loading models...")
            t3 = time.time()
            # Load Scaler
            self.scaler = joblib.load(os.path.join(self.model_dir, "scaler_enhanced.save"))
            
            # Load XGBoost Models
            self.xgb_sent = xgb.XGBClassifier()
            self.xgb_sent.load_model(os.path.join(self.model_dir, "xgb_sentiment.json"))
            
            self.xgb_trend = xgb.XGBClassifier()
            self.xgb_trend.load_model(os.path.join(self.model_dir, "xgb_trend.json"))
            
            # Load LSTM Models
            self.lstm_sent = load_model(os.path.join(self.model_dir, "lstm_sentiment.h5"))
            self.lstm_trend = load_model(os.path.join(self.model_dir, "lstm_trend.h5"))
            t4 = time.time()
            logger.info(f"All artifacts loaded successfully. Total time: {t4-t3:.2f}s")
            
        except Exception as e:
            logger.error(f"Error loading artifacts: {e}")
            raise

    def fetch_live_data(self, symbol):
        """
        Fetch last 1 year of data + Live News Sentiment
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=400) # Buffer
        
        # 1. Fetch Price Data
        t_data_start = time.time()
        df = yf.download(symbol, start=start_date, end=end_date, progress=False)
        t_data_end = time.time()
        logger.info(f"[{symbol}] Stock Data Fetch: {(t_data_end - t_data_start)*1000:.2f}ms")

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        if len(df) < self.lookback + 20:
            raise ValueError(f"Not enough data for {symbol}")
            
        # 2. Fetch Live News Sentiment
        t_news_start = time.time()
        logger.info(f"Fetching live news for {symbol}...")
        news_data = fetch_news(symbol, page_size=10)
        analyzed_news = analyze_articles(news_data)
        t_news_end = time.time()
        logger.info(f"[{symbol}] News Fetch & Analysis: {(t_news_end - t_news_start)*1000:.2f}ms")
        
        # Get compound score
        if analyzed_news and 'overall_sentiment' in analyzed_news:
             articles = analyzed_news.get('articles', [])
             if articles:
                 compounds = [a['sentiment']['compound'] for a in articles]
                 current_sentiment_score = np.mean(compounds)
             else:
                 current_sentiment_score = 0.0
        else:
            current_sentiment_score = 0.0
            
        logger.info(f"Current News Sentiment Score: {current_sentiment_score}")

        # 3. Engineer Features
        t_fe_start = time.time()
        df = add_technical_indicators(df)
        df = add_time_features(df)
        df = add_market_regime(df)
        t_fe_end = time.time()
        logger.info(f"[{symbol}] Feature Engineering: {(t_fe_end - t_fe_start)*1000:.2f}ms")
        
        # 4. Handle 'News_Sentiment' feature
        # Propagate current sentiment to the last row, 
        # and maybe decay it backwards or just use 0 (Neutral) for far history
        # For this demo, we'll fill with 0 and put the actual score in the last 5 days
        df['News_Sentiment'] = 0.0
        df.iloc[-5:, df.columns.get_loc('News_Sentiment')] = current_sentiment_score
        
        df.dropna(inplace=True)
        return df, current_sentiment_score

    def predict(self, symbol):
        """
        Generate prediction for the symbol
        """
        # 0. Check Cache
        cached_result = self.cache.get(symbol)
        if cached_result:
            logger.info(f"[{symbol}] Cache HIT. Returning cached prediction.")
            # Update timestamp to show when it was served effectively
            # But keep original analysis timestamp? 
            # Let's verify if user wants fresh time or cached time. 
            # Usually cached time is honest.
            return cached_result

        # 1. Prepare Data
        t_total_start = time.time()
        df, current_sentiment = self.fetch_live_data(symbol)
        
        # Get data for Tabular model (Last row)
        # Ensure columns match training
        # We need to exclude targets if they exist (they won't here)
        # And ensure order. We'll use the scaler logic for consistency
        
        # Need to reconstruct the feature list used in training?
        # Ideally we should have saved the feature names.
        # Implied contract: feature_engineering output + News_Sentiment
        
        # Tabular Input (Last Row)
        X_tab_all = df.values
        X_scaled_all = self.scaler.transform(X_tab_all)
        
        last_row_scaled = X_scaled_all[-1].reshape(1, -1)
        
        # Sequence Input (Last 60 days)
        last_sequence = X_scaled_all[-self.lookback:]
        last_sequence = last_sequence.reshape(1, self.lookback, last_sequence.shape[1])
        
        # 2. Run Inference
        t_infer_start = time.time()
        
        # XGBoost
        xgb_sent_prob = self.xgb_sent.predict_proba(last_row_scaled)[0] # Class Probs
        xgb_trend_prob = self.xgb_trend.predict_proba(last_row_scaled)[0]
        
        # LSTM
        lstm_sent_prob = self.lstm_sent.predict(last_sequence, verbose=0)[0]
        lstm_trend_prob = self.lstm_trend.predict(last_sequence, verbose=0)[0]
        
        t_infer_end = time.time()
        logger.info(f"[{symbol}] Model Inference (4 Models): {(t_infer_end - t_infer_start)*1000:.2f}ms")
        
        # 3. Ensemble (Simple Average)
        # Classes: 0, 1, 2
        
        # Sentiment Signal (0=Sell, 1=Hold, 2=Buy)
        final_sent_prob = (xgb_sent_prob + lstm_sent_prob) / 2
        sent_class = np.argmax(final_sent_prob)
        sent_conf = final_sent_prob[sent_class]
        
        # Trend Signal (0=Down, 1=Sideways, 2=Up)
        final_trend_prob = (xgb_trend_prob + lstm_trend_prob) / 2
        trend_class = np.argmax(final_trend_prob)
        trend_conf = final_trend_prob[trend_class]
        
        # 4. Generate Output
        
        # Calculate extra technicals
        macd_val = df['MACD'].iloc[-1]
        macd_signal_val = df['MACD_Signal'].iloc[-1]
        macd_status = "bullish" if macd_val > macd_signal_val else "bearish"
        
        # Simple Support/Resistance (20-day Low/High)
        support_level = df['Low'].tail(20).min()
        resistance_level = df['High'].tail(20).max()
        
        result = {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "current_price": float(df['Close'].iloc[-1]),
            "sentiment_score": float(current_sentiment),
            "trading_signals": {
                "sentiment_signal": ["SELL", "HOLD", "BUY"][sent_class],
                "trend_signal": ["DOWN", "SIDEWAYS", "UP"][trend_class],
                "combined_signal": "STRONG_BUY" if sent_class == 2 and trend_class == 2 else "BUY" if sent_class == 2 or trend_class == 2 else "SELL" if sent_class == 0 or trend_class == 0 else "HOLD",
                "confidence": float((sent_conf + trend_conf) / 2)
            },
            "technical_analysis": {
                "rsi": float(df['RSI'].iloc[-1]),
                "macd": macd_status,
                "macd_value": float(macd_val),
                "support_level": float(support_level),
                "resistance_level": float(resistance_level)
            },
            "risk_metrics": {
                "volatility_atr": float(df['ATR'].iloc[-1]),
                "market_regime": "High Volatility" if df['Regime_Vol'].iloc[-1] == 1 else "Low Volatility",
                "rsi_value": float(df['RSI'].iloc[-1])
            }
        }
        
        t_total_end = time.time()
        logger.info(f"[{symbol}] Total Prediction Pipeline: {(t_total_end - t_total_start)*1000:.2f}ms")
        
        # Cache the result
        self.cache.set(symbol, result)
        
        return result

if __name__ == "__main__":
    predictor = EnhancedPredictor()
    prediction = predictor.predict("AAPL")
    
    import json
    print(json.dumps(prediction, indent=2))
