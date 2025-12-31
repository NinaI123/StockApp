import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
import joblib
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime, timedelta

def prepare_data(symbol="AAPL", lookback=60, test_size=0.2):
    # Get more data (3 years) and multiple features
    end_date = datetime.now()
    start_date = end_date - timedelta(days=3*365)
    
    df = yf.download(symbol, start=start_date, end=end_date)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    
    # Add technical indicators
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    df['RSI'] = compute_rsi(df['Close'])
    
    # Drop rows with NaN values (from moving averages and RSI)
    df = df.dropna()
    
    # Scale data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)
    
    # Create sequences
    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i-lookback:i])
        y.append(scaled_data[i, 3])  # Close price is at index 3
    
    X, y = np.array(X), np.array(y)
    
    # Split data
    split = int((1 - test_size) * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    return X_train, X_test, y_train, y_test, scaler, df

def compute_rsi(prices, window=14):
    """
    Calculate Relative Strength Index (RSI)
    Args:
        prices: Pandas Series of closing prices
        window: Lookback window for RSI calculation
    Returns:
        Pandas Series with RSI values
    """
    deltas = prices.diff()
    
    # Get positive gains (up) and negative gains (down)
    up = deltas.clip(lower=0)
    down = -1 * deltas.clip(upper=0)
    
    # Calculate simple averages
    avg_up = up.rolling(window).mean()
    avg_down = down.rolling(window).mean()
    
    # Calculate Relative Strength
    rs = avg_up / avg_down
    
    # Calculate RSI
    rsi = 100 - (100 / (1 + rs))
    
    # The first 'window' days will have NaN values
    return rsi
def build_model(input_shape):
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        LSTM(64, return_sequences=True),
        Dropout(0.3),
        LSTM(32),
        Dropout(0.3),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def train_model(symbol="AAPL"):
    # Prepare data
    X_train, X_test, y_train, y_test, scaler, df = prepare_data(symbol)
    
    # Build model
    model = build_model((X_train.shape[1], X_train.shape[2]))
    
    # Early stopping
    early_stop = EarlyStopping(monitor='val_loss', patience=10)
    
    # Train
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[early_stop],
        verbose=1
    )
    
    # Save model
    os.makedirs("models/saved", exist_ok=True)
    model.save("models/saved/lstm_model.h5")
    joblib.dump(scaler, "models/saved/scaler.save")
    
    # Evaluate
    predictions = model.predict(X_test)
    y_true = scaler.inverse_transform(
        np.concatenate(
            (y_test.reshape(-1,1), np.zeros((len(y_test), df.shape[1]-1))),
            axis=1
        )
    )[:, 0]
    y_pred = scaler.inverse_transform(
        np.concatenate(
            (predictions.reshape(-1,1), np.zeros((len(predictions), df.shape[1]-1))),
            axis=1
        )
    )[:, 0]
    
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    print(f"MAE: {mae:.2f}")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R² Score: {r2:.2f}")
    
    return model, scaler

if __name__ == "__main__":
    train_model()