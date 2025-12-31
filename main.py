#system arhcitecture changed to FASTAPI, before it was flask
import os
import sqlite3
import logging
import json
from datetime import datetime
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, Request, HTTPException, Depends, File, UploadFile, Query, status
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import pandas as pd
import yfinance as yf

# Existing utilities and models
# Ensure these files exist and are importable
from utils.sentiment import analyze_articles
from utils.news_api import fetch_news
from utils.recommendations import get_recommendations
from utils.stock_utils import get_stock_summary, fetch_historical_data
from models.predict_enhanced import EnhancedPredictor

# --- Configuration & Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data/app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

DB_FILE = 'data/portfolio.db'
os.makedirs('data', exist_ok=True)

app = FastAPI(
    title="AI Stock Analyzer",
    description="Stock Analysis API with Sentiment and Trend Prediction",
    version="2.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Templates & Static
templates = Jinja2Templates(directory="templates")
# app.mount("/static", StaticFiles(directory="static"), name="static") # Uncomment if static exists

# Initialize Predictor
try:
    predictor = EnhancedPredictor()
    logger.info("Enhanced Predictor initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize predictor: {e}")
    predictor = None

# --- Database Dependency ---
def get_db():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

def init_db():
    """Initialize the database with required tables"""
    # Use a direct connection here since we are not in a request scope
    conn = sqlite3.connect(DB_FILE)
    try:
        conn.execute('''
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            qty REAL NOT NULL,
            price REAL NOT NULL,
            trade_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            notes TEXT
        )
        ''')
        conn.execute('''
        CREATE TABLE IF NOT EXISTS watchlist (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL UNIQUE,
            added_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        conn.commit()
    finally:
        conn.close()

# Run init on startup
init_db()

# --- Pydantic Models ---
class TradeCreate(BaseModel):
    symbol: str
    qty: float
    price: float
    notes: Optional[str] = ""

class WatchlistAdd(BaseModel):
    symbol: str

class BacktestRequest(BaseModel):
    symbol: str = "AAPL"
    period: str = "1y"
    strategy: str = "trend_following"

# --- Routes ---

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the main application page"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/stock/summary")
async def get_stock_summary_route(symbol: str = Query(..., min_length=1, max_length=10), period: str = "1y"):
    """Get summary information for a stock"""
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(period=period)
        
        if hist.empty:
            raise HTTPException(status_code=404, detail="No data available for this symbol")
        
        summary = get_stock_summary(stock, hist)
        return summary
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching summary: {e}")
        # Check for yfinance specific errors if needed
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stock/historical")
async def get_historical_data_route(symbol: str = Query(..., min_length=1, max_length=10), period: str = "1y"):
    """Get historical price data for a stock"""
    try:
        data = fetch_historical_data(symbol, period)
        return data
    except Exception as e:
        logger.error(f"Error fetching historical data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stock/analysis")
async def get_stock_analysis(symbol: str = Query("AAPL", min_length=1, max_length=10)):
    """Get comprehensive stock analysis"""
    if not predictor:
        raise HTTPException(status_code=503, detail="Predictor not initialized")
    
    symbol = symbol.upper()
    try:
        analysis = predictor.predict(symbol)
        
        # Construct Response
        response = {
            "symbol": analysis.get("symbol", symbol),
            "current_price": analysis.get("current_price", 0.0),
            "sentiment_score": analysis.get("sentiment_score", 0.0),
            "trading_signals": analysis["trading_signals"],
            "technical_analysis": analysis["technical_analysis"],
            "sentiment_analysis": {
                "avg_sentiment": analysis["sentiment_score"],
                "positive_articles": 0, 
                "negative_articles": 0,
                "key_topics": [] 
            },
            "risk_assessment": {
                "volatility": "high" if analysis["risk_metrics"]["market_regime"] == "High Volatility" else "medium",
                "market_regime": analysis["risk_metrics"]["market_regime"],
                "recommended_position_size": "small" if analysis["risk_metrics"]["volatility_atr"] > 5 else "moderate"
            }
        }
        return response
    except Exception as e:
        logger.error(f"Analysis failed for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/api/backtest")
async def backtest_strategy_route(request: BacktestRequest):
    """Test strategy performance (Stub/Simulated)"""
    # Simply return the stub data as requested
    return {
        "symbol": request.symbol,
        "period": request.period,
        "strategy": request.strategy,
        "performance": {
            "total_return": "15.4%",
            "annualized_return": "12.1%",
            "max_drawdown": "-8.5%",
            "sharpe_ratio": 1.45,
            "trades_executed": 24,
            "win_rate": "62%"
        },
        "benchmark_return": "10.2%"
    }

@app.get("/api/model/performance")
async def get_model_performance():
    """Get model accuracy metrics"""
    metrics_path = "models/saved/metrics.json"
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            return metrics
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load metrics: {e}")
    else:
        raise HTTPException(status_code=404, detail="Metrics not found")

@app.get("/api/insights/daily")
async def get_daily_insights():
    """Get daily market insights"""
    proxy_symbol = "SPY"
    if predictor:
        try:
            analysis = predictor.predict(proxy_symbol)
            trend = analysis["trading_signals"]["trend_signal"]
            return {
                "date": datetime.now().isoformat(),
                "market_sentiment": analysis["trading_signals"]["sentiment_signal"],
                "trend": trend,
                "summary": f"The market is currently showing a {trend} trend with {analysis['trading_signals']['sentiment_signal']} sentiment.",
                "top_movers": [
                    {"symbol": "NVDA", "change": "+2.5%"},
                    {"symbol": "TSLA", "change": "-1.2%"},
                    {"symbol": "AMD", "change": "+1.8%"}
                ]
            }
        except Exception:
            raise HTTPException(status_code=503, detail="Market insights unavailable")
    raise HTTPException(status_code=503, detail="Predictor maintenance")

@app.get("/api/news")
async def get_news_route(symbol: str = Query(..., min_length=1), limit: int = 10):
    """Get news articles"""
    limit = min(limit, 20)
    logger.info(f"Fetching news for: {symbol}")
    news = fetch_news(symbol, page_size=limit)
    analyzed = analyze_articles(news)
    return analyzed

# --- Portfolio Routes ---

@app.get("/api/portfolio/trades")
async def get_trades(db: sqlite3.Connection = Depends(get_db)):
    """View all trades"""
    cursor = db.execute("SELECT * FROM trades ORDER BY trade_date DESC")
    trades = cursor.fetchall()
    return [dict(trade) for trade in trades]

@app.post("/api/portfolio/trades", status_code=status.HTTP_201_CREATED)
async def add_trade(trade: TradeCreate, db: sqlite3.Connection = Depends(get_db)):
    """Add a new trade"""
    try:
        db.execute(
            "INSERT INTO trades (symbol, qty, price, notes) VALUES (?, ?, ?, ?)",
            (trade.symbol, trade.qty, trade.price, trade.notes)
        )
        db.commit()
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to save trade: {e}")

@app.get("/api/portfolio/watchlist")
async def get_watchlist(db: sqlite3.Connection = Depends(get_db)):
    """View watchlist"""
    cursor = db.execute("SELECT * FROM watchlist ORDER BY added_date DESC")
    watchlist = cursor.fetchall()
    return [dict(item) for item in watchlist]

@app.post("/api/portfolio/watchlist", status_code=status.HTTP_201_CREATED)
async def add_to_watchlist(item: WatchlistAdd, db: sqlite3.Connection = Depends(get_db)):
    """Add to watchlist"""
    try:
        db.execute(
            "INSERT OR IGNORE INTO watchlist (symbol) VALUES (?)",
            (item.symbol,)
        )
        db.commit()
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to add to watchlist: {e}")

@app.delete("/api/portfolio/watchlist")
async def remove_from_watchlist(item: WatchlistAdd, db: sqlite3.Connection = Depends(get_db)):
    """Remove from watchlist"""
    try:
        db.execute(
            "DELETE FROM watchlist WHERE symbol = ?",
            (item.symbol,)
        )
        db.commit()
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to remove from watchlist: {e}")

@app.post("/api/recommend")
async def recommend_route(file: UploadFile = File(...)):
    """Generate recommendations from CSV"""
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")
    
    try:
        df = pd.read_csv(file.file)
        recommendations = get_recommendations(df)
        return recommendations
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process file: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=True)
