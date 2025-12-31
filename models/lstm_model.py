import os
import numpy as np
import pandas as pd
from keras.models import load_model
import joblib
from datetime import datetime, timedelta
import yfinance as yf
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def predict_stock_prices(symbol, days_to_predict=30):
    """Predict stock prices with enhanced error handling"""
    try:
        # Verify inputs
        if not symbol or not isinstance(symbol, str):
            raise ValueError("Invalid stock symbol")
            
        if days_to_predict <= 0 or days_to_predict > 90:
            raise ValueError("Prediction days must be between 1-90")
        
        # Load model and scaler with verification
        model_dir = "models/saved"
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
            
        model_path = os.path.join(model_dir, "lstm_model.h5")
        scaler_path = os.path.join(model_dir, "scaler.save")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
        
        logger.info(f"Loading model and scaler for {symbol}")
        model = load_model(model_path)
        scaler = joblib.load(scaler_path)
        
        # Fetch historical data
        logger.info(f"Fetching historical data for {symbol}")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        df = yf.download(symbol, start=start_date, end=end_date)
        
        if df.empty:
            raise ValueError(f"No data available for {symbol}")
            
        # Prepare data for prediction
        scaled_data = scaler.transform(df[['Close']])
        
        # Generate prediction
        last_sequence = scaled_data[-60:]  # Using last 60 days for prediction
        predictions = []
        
        for _ in range(days_to_predict):
            # Reshape input for model
            X = last_sequence[-60:].reshape(1, 60, 1)
            
            # Predict next day
            pred = model.predict(X)[0][0]
            predictions.append(pred)
            
            # Update sequence
            last_sequence = np.append(last_sequence, pred).reshape(-1, 1)
        
        # Inverse transform predictions
        predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        
        # Prepare dates
        last_date = df.index[-1]
        future_dates = [last_date + timedelta(days=i) for i in range(1, days_to_predict+1)]
        
        return {
            "historical": {
                "dates": df.index.strftime('%Y-%m-%d').tolist(),
                "prices": df['Close'].tolist()
            },
            "predicted": {
                "dates": [d.strftime('%Y-%m-%d') for d in future_dates],
                "prices": [float(p[0]) for p in predictions]
            }
        }
        
    except Exception as e:
        logger.error(f"Prediction failed for {symbol}: {str(e)}")
        raise  # Re-raise for Flask to handle