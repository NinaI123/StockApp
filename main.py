"""
main.py — Fantasy Finance FastAPI Backend (v2.0)
Migrated from SQLite → PostgreSQL. Added auth, predictions, profile, and new Wars endpoints.
"""
import os
import json
import time
import random
import logging
from datetime import datetime, date, timedelta
from typing import List, Optional, Dict, Any

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Request, HTTPException, Depends, File, UploadFile, Query, status
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
import pandas as pd
import yfinance as yf

from utils.sentiment import analyze_articles
from utils.news_api import fetch_news
from utils.recommendations import get_recommendations
from utils.stock_utils import get_stock_summary, fetch_historical_data
from models.predict_enhanced import EnhancedPredictor
from db import get_db, init_db
from auth import (
    hash_password, verify_password, create_access_token,
    get_current_user, get_optional_user
)

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Fantasy Finance API",
    description="AI-Powered Competitive Stock Platform",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize DB and predictor on startup
init_db()

try:
    predictor = EnhancedPredictor()
    logger.info("Enhanced Predictor initialized.")
except Exception as e:
    logger.error(f"Predictor init failed: {e}")
    predictor = None

# ── Score cache ───────────────────────────────────────────────────────────────
_score_cache: Dict[int, tuple] = {}
_SCORE_CACHE_TTL = 60  # seconds

# ── Pydantic Models ───────────────────────────────────────────────────────────
class UserSignup(BaseModel):
    email: str
    password: str

class UserLogin(BaseModel):
    email: str
    password: str

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

class LeagueCreate(BaseModel):
    name: str
    is_public: bool = True
    max_teams: int = 8
    season_weeks: int = 2

class TeamCreate(BaseModel):
    league_code: str
    player_name: str
    team_name: str
    stocks: List[str]

class MatchupCreate(BaseModel):
    league_id: int
    team1_id: int
    team2_id: int

class MessageCreate(BaseModel):
    league_id: int
    player_name: str
    message: str

class DraftPick(BaseModel):
    league_id: int
    team_id: int
    symbol: str
    pick_round: int

class PredictionCreate(BaseModel):
    symbol: str
    target_price: float
    resolution_date: str  # ISO date string YYYY-MM-DD

# ── Helpers ───────────────────────────────────────────────────────────────────
def _current_week() -> tuple:
    today = date.today()
    monday = today - timedelta(days=today.weekday())
    friday = monday + timedelta(days=4)
    return monday.isoformat(), friday.isoformat()

def _gen_code(db, table: str) -> str:
    for _ in range(30):
        code = str(random.randint(100000, 999999))
        cur = db.cursor()
        cur.execute(f"SELECT id FROM {table} WHERE code = %s", (code,))
        if not cur.fetchone():
            return code
    raise HTTPException(status_code=500, detail="Could not generate unique code.")

def _compute_streak(match_results: list) -> str:
    if not match_results:
        return "-"
    last = match_results[-1]
    count = 0
    for r in reversed(match_results):
        if r == last:
            count += 1
        else:
            break
    return f"🔥{count}W" if last else f"{count}L"

# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# ════════════════════════════════════════════════════════════════════
# 🔐 AUTH
# ════════════════════════════════════════════════════════════════════

@app.post("/api/auth/signup", status_code=201)
async def signup(body: UserSignup, db=Depends(get_db)):
    cur = db.cursor()
    cur.execute("SELECT id FROM users WHERE email = %s", (body.email.lower(),))
    if cur.fetchone():
        raise HTTPException(status_code=400, detail="Email already registered.")
    if len(body.password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters.")
    hashed = hash_password(body.password)
    cur.execute(
        "INSERT INTO users (email, password_hash) VALUES (%s, %s) RETURNING id",
        (body.email.lower(), hashed)
    )
    user_id = cur.fetchone()["id"]
    token = create_access_token({"sub": str(user_id), "email": body.email.lower()})
    return {"access_token": token, "token_type": "bearer", "user_id": user_id}


@app.post("/api/auth/login")
async def login(body: UserLogin, db=Depends(get_db)):
    cur = db.cursor()
    cur.execute("SELECT id, password_hash FROM users WHERE email = %s", (body.email.lower(),))
    user = cur.fetchone()
    if not user or not verify_password(body.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email or password.")
    token = create_access_token({"sub": str(user["id"]), "email": body.email.lower()})
    return {"access_token": token, "token_type": "bearer", "user_id": user["id"]}


# ════════════════════════════════════════════════════════════════════
# 📈 STOCK ANALYSIS
# ════════════════════════════════════════════════════════════════════

@app.get("/api/stock/summary")
async def get_stock_summary_route(
    symbol: str = Query(..., min_length=1, max_length=10),
    period: str = "1y"
):
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(period=period)
        if hist.empty:
            raise HTTPException(status_code=404, detail="No data for this symbol.")
        return get_stock_summary(stock, hist)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stock/historical")
async def get_historical_data_route(
    symbol: str = Query(..., min_length=1, max_length=10),
    period: str = "1y"
):
    try:
        return fetch_historical_data(symbol, period)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stock/analysis")
async def get_stock_analysis(symbol: str = Query("AAPL", min_length=1, max_length=10)):
    if not predictor:
        raise HTTPException(status_code=503, detail="Predictor not initialized.")
    symbol = symbol.upper()
    try:
        analysis = predictor.predict(symbol)
        return {
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
    except Exception as e:
        logger.error(f"Analysis failed for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.get("/api/news")
async def get_news_route(symbol: str = Query(..., min_length=1), limit: int = 10):
    limit = min(limit, 20)
    news = fetch_news(symbol, page_size=limit)
    return analyze_articles(news)


@app.post("/api/backtest")
async def backtest_strategy_route(request: BacktestRequest):
    return {
        "symbol": request.symbol, "period": request.period, "strategy": request.strategy,
        "performance": {
            "total_return": "15.4%", "annualized_return": "12.1%",
            "max_drawdown": "-8.5%", "sharpe_ratio": 1.45,
            "trades_executed": 24, "win_rate": "62%"
        },
        "benchmark_return": "10.2%"
    }


@app.get("/api/model/performance")
async def get_model_performance():
    metrics_path = "models/saved/metrics.json"
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            return json.load(f)
    raise HTTPException(status_code=404, detail="Metrics not found.")


@app.get("/api/insights/daily")
async def get_daily_insights():
    if not predictor:
        raise HTTPException(status_code=503, detail="Predictor unavailable.")
    try:
        analysis = predictor.predict("SPY")
        trend = analysis["trading_signals"]["trend_signal"]
        return {
            "date": datetime.now().isoformat(),
            "market_sentiment": analysis["trading_signals"]["sentiment_signal"],
            "trend": trend,
            "summary": f"Market showing {trend} trend with {analysis['trading_signals']['sentiment_signal']} sentiment.",
            "top_movers": [
                {"symbol": "NVDA", "change": "+2.5%"},
                {"symbol": "TSLA", "change": "-1.2%"},
                {"symbol": "AMD", "change": "+1.8%"}
            ]
        }
    except Exception:
        raise HTTPException(status_code=503, detail="Market insights unavailable.")


@app.post("/api/recommend")
async def recommend_route(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files supported.")
    df = pd.read_csv(file.file)
    return get_recommendations(df)


# ════════════════════════════════════════════════════════════════════
# 💼 PORTFOLIO
# ════════════════════════════════════════════════════════════════════

@app.get("/api/portfolio/trades")
async def get_trades(db=Depends(get_db), user=Depends(get_optional_user)):
    cur = db.cursor()
    if user:
        cur.execute("SELECT * FROM trades WHERE user_id = %s ORDER BY trade_date DESC", (int(user["sub"]),))
    else:
        cur.execute("SELECT * FROM trades ORDER BY trade_date DESC LIMIT 50")
    return [dict(r) for r in cur.fetchall()]


@app.post("/api/portfolio/trades", status_code=201)
async def add_trade(trade: TradeCreate, db=Depends(get_db), user=Depends(get_optional_user)):
    cur = db.cursor()
    user_id = int(user["sub"]) if user else None
    cur.execute(
        "INSERT INTO trades (user_id, symbol, qty, price, notes) VALUES (%s, %s, %s, %s, %s)",
        (user_id, trade.symbol.upper(), trade.qty, trade.price, trade.notes)
    )
    return {"status": "success"}


@app.get("/api/portfolio/watchlist")
async def get_watchlist(db=Depends(get_db), user=Depends(get_optional_user)):
    cur = db.cursor()
    if user:
        cur.execute("SELECT * FROM watchlist WHERE user_id = %s ORDER BY added_date DESC", (int(user["sub"]),))
    else:
        cur.execute("SELECT * FROM watchlist ORDER BY added_date DESC LIMIT 50")
    return [dict(r) for r in cur.fetchall()]


@app.post("/api/portfolio/watchlist", status_code=201)
async def add_to_watchlist(item: WatchlistAdd, db=Depends(get_db), user=Depends(get_optional_user)):
    cur = db.cursor()
    user_id = int(user["sub"]) if user else None
    cur.execute(
        "INSERT INTO watchlist (user_id, symbol) VALUES (%s, %s) ON CONFLICT (user_id, symbol) DO NOTHING",
        (user_id, item.symbol.upper())
    )
    return {"status": "success"}


@app.delete("/api/portfolio/watchlist")
async def remove_from_watchlist(item: WatchlistAdd, db=Depends(get_db), user=Depends(get_optional_user)):
    cur = db.cursor()
    if user:
        cur.execute("DELETE FROM watchlist WHERE user_id = %s AND symbol = %s", (int(user["sub"]), item.symbol.upper()))
    else:
        cur.execute("DELETE FROM watchlist WHERE symbol = %s", (item.symbol.upper(),))
    return {"status": "success"}


# ════════════════════════════════════════════════════════════════════
# ⚔️  PORTFOLIO WARS  ──  /api/wars/*
# ════════════════════════════════════════════════════════════════════

@app.post("/api/wars/league", status_code=201)
async def create_league(body: LeagueCreate, db=Depends(get_db)):
    week_start, week_end = _current_week()
    code = _gen_code(db, "pw_leagues")
    cur = db.cursor()
    cur.execute(
        "INSERT INTO pw_leagues (name, is_public, max_teams, season_weeks, week_start, week_end, code) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING id",
        (body.name.strip(), body.is_public, body.max_teams, body.season_weeks, week_start, week_end, code)
    )
    row = cur.fetchone()
    return {"id": row["id"], "code": code, "name": body.name,
            "week_start": week_start, "week_end": week_end}


@app.post("/api/wars/team", status_code=201)
async def join_league(body: TeamCreate, db=Depends(get_db), user=Depends(get_optional_user)):
    if not (3 <= len(body.stocks) <= 5):
        raise HTTPException(status_code=400, detail="Team must have 3–5 stocks.")
    symbols = [s.upper().strip() for s in body.stocks]
    cur = db.cursor()
    cur.execute("SELECT id FROM pw_leagues WHERE code = %s", (body.league_code.strip(),))
    league = cur.fetchone()
    if not league:
        raise HTTPException(status_code=404, detail=f"League code '{body.league_code}' not found.")
    team_code = _gen_code(db, "pw_teams")
    user_id = int(user["sub"]) if user else None
    cur.execute(
        "INSERT INTO pw_teams (league_id, user_id, player_name, team_name, stocks, code) "
        "VALUES (%s, %s, %s, %s, %s, %s) RETURNING id",
        (league["id"], user_id, body.player_name.strip(), body.team_name.strip(), json.dumps(symbols), team_code)
    )
    row = cur.fetchone()
    return {"id": row["id"], "code": team_code, "league_id": league["id"],
            "player_name": body.player_name, "team_name": body.team_name, "stocks": symbols}


@app.get("/api/wars/leagues/public")
async def list_public_leagues(
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1, le=50),
    db=Depends(get_db)
):
    offset = (page - 1) * limit
    cur = db.cursor()
    cur.execute(
        "SELECT l.*, COUNT(t.id) AS team_count FROM pw_leagues l "
        "LEFT JOIN pw_teams t ON t.league_id = l.id "
        "WHERE l.is_public = TRUE "
        "GROUP BY l.id ORDER BY l.created_at DESC LIMIT %s OFFSET %s",
        (limit, offset)
    )
    return [dict(r) for r in cur.fetchall()]


@app.get("/api/wars/league/{league_ref}")
async def get_league(league_ref: str, db=Depends(get_db)):
    cur = db.cursor()
    if league_ref.isdigit():
        cur.execute("SELECT * FROM pw_leagues WHERE id = %s", (int(league_ref),))
    else:
        cur.execute("SELECT * FROM pw_leagues WHERE code = %s", (league_ref,))
    league = cur.fetchone()
    if not league:
        raise HTTPException(status_code=404, detail="League not found.")
    league_id = league["id"]

    cur.execute("SELECT * FROM pw_teams WHERE league_id = %s ORDER BY created_at", (league_id,))
    teams_rows = cur.fetchall()

    teams = []
    for t in teams_rows:
        cur.execute(
            "SELECT r.winner_id FROM pw_results r "
            "JOIN pw_matchups m ON r.matchup_id = m.id "
            "WHERE (m.team1_id = %s OR m.team2_id = %s) AND m.league_id = %s "
            "ORDER BY r.scored_at",
            (t["id"], t["id"], league_id)
        )
        results = cur.fetchall()
        match_bools = [r["winner_id"] == t["id"] for r in results]
        wins = sum(match_bools)
        losses = len(match_bools) - wins
        streak = _compute_streak(match_bools)
        teams.append({
            "id": t["id"], "code": t["code"],
            "player_name": t["player_name"],
            "team_name": t["team_name"] or t["player_name"],
            "stocks": json.loads(t["stocks"]),
            "wins": wins, "losses": losses, "streak": streak,
        })
    teams.sort(key=lambda x: (-x["wins"], x["losses"]))

    cur.execute(
        "SELECT m.*, r.team1_return, r.team2_return, r.winner_id AS result_winner "
        "FROM pw_matchups m LEFT JOIN pw_results r ON r.matchup_id = m.id "
        "WHERE m.league_id = %s ORDER BY m.id",
        (league_id,)
    )
    matchups = [dict(m) for m in cur.fetchall()]
    return {
        "id": league["id"], "code": league["code"], "name": league["name"],
        "week_start": league["week_start"], "week_end": league["week_end"],
        "is_public": league["is_public"],
        "teams": teams, "matchups": matchups,
    }


@app.post("/api/wars/matchup", status_code=201)
async def create_matchup(body: MatchupCreate, db=Depends(get_db)):
    if body.team1_id == body.team2_id:
        raise HTTPException(status_code=400, detail="A team cannot play itself.")
    cur = db.cursor()
    for tid in (body.team1_id, body.team2_id):
        cur.execute("SELECT id FROM pw_teams WHERE id = %s AND league_id = %s", (tid, body.league_id))
        if not cur.fetchone():
            raise HTTPException(status_code=404, detail=f"Team {tid} not in league {body.league_id}.")
    week_start, _ = _current_week()
    cur.execute(
        "INSERT INTO pw_matchups (league_id, team1_id, team2_id, week_start) "
        "VALUES (%s, %s, %s, %s) RETURNING id",
        (body.league_id, body.team1_id, body.team2_id, week_start)
    )
    row = cur.fetchone()
    return {"id": row["id"], "league_id": body.league_id,
            "team1_id": body.team1_id, "team2_id": body.team2_id, "week_start": week_start}


@app.post("/api/wars/score/{matchup_id}")
async def score_matchup(matchup_id: int, db=Depends(get_db)):
    cached = _score_cache.get(matchup_id)
    if cached and (time.time() - cached[0]) < _SCORE_CACHE_TTL:
        return cached[1]

    cur = db.cursor()
    cur.execute("SELECT * FROM pw_matchups WHERE id = %s", (matchup_id,))
    matchup = cur.fetchone()
    if not matchup:
        raise HTTPException(status_code=404, detail="Matchup not found.")

    cur.execute("SELECT * FROM pw_teams WHERE id = %s", (matchup["team1_id"],))
    t1 = cur.fetchone()
    cur.execute("SELECT * FROM pw_teams WHERE id = %s", (matchup["team2_id"],))
    t2 = cur.fetchone()

    stocks1 = json.loads(t1["stocks"])
    stocks2 = json.loads(t2["stocks"])
    all_stocks = list(set(stocks1 + stocks2))

    try:
        week_start = str(matchup["week_start"])
        week_end_dt = date.fromisoformat(week_start) + timedelta(days=7)
        raw = yf.download(
            all_stocks, start=week_start, end=week_end_dt.isoformat(),
            progress=False, auto_adjust=True
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"yfinance fetch failed: {e}")

    def calc_return(symbol: str) -> Optional[float]:
        try:
            closes = raw["Close"][symbol] if len(all_stocks) > 1 else raw["Close"]
            closes = closes.dropna()
            if len(closes) < 2:
                return None
            return float((closes.iloc[-1] - closes.iloc[0]) / closes.iloc[0] * 100)
        except Exception:
            return None

    def team_score(symbols: List[str]) -> tuple:
        returns = {s: calc_return(s) for s in symbols}
        valid = [v for v in returns.values() if v is not None]
        avg = round(sum(valid) / len(valid), 4) if valid else 0.0
        return avg, {s: (round(r, 4) if r is not None else "N/A") for s, r in returns.items()}

    score1, breakdown1 = team_score(stocks1)
    score2, breakdown2 = team_score(stocks2)

    if score1 > score2:
        winner_id, winner_name = t1["id"], t1["player_name"]
    elif score2 > score1:
        winner_id, winner_name = t2["id"], t2["player_name"]
    else:
        winner_id, winner_name = None, "TIE"

    cur.execute(
        "INSERT INTO pw_results (matchup_id, team1_return, team2_return, winner_id) "
        "VALUES (%s, %s, %s, %s) "
        "ON CONFLICT (matchup_id) DO UPDATE SET "
        "team1_return = EXCLUDED.team1_return, team2_return = EXCLUDED.team2_return, "
        "winner_id = EXCLUDED.winner_id, scored_at = NOW()",
        (matchup_id, score1, score2, winner_id)
    )

    result = {
        "matchup_id": matchup_id,
        "week_start": str(matchup["week_start"]),
        "team1": {"id": t1["id"], "player": t1["player_name"], "stocks": stocks1,
                  "avg_return_pct": score1, "breakdown": breakdown1},
        "team2": {"id": t2["id"], "player": t2["player_name"], "stocks": stocks2,
                  "avg_return_pct": score2, "breakdown": breakdown2},
        "winner": winner_name,
        "winner_id": winner_id,
    }
    _score_cache[matchup_id] = (time.time(), result)
    return result


@app.get("/api/wars/matchups/{matchup_id}/report")
async def get_matchup_report(matchup_id: int, db=Depends(get_db)):
    """Return cached AI post-game report. Stub until Claude API key is added."""
    cur = db.cursor()
    cur.execute("SELECT * FROM pw_ai_reports WHERE matchup_id = %s", (matchup_id,))
    cached = cur.fetchone()
    if cached:
        return {"matchup_id": matchup_id, "report": cached["report_text"],
                "generated_at": str(cached["generated_at"])}

    # Stub report — Claude API will replace this
    cur.execute("SELECT * FROM pw_results WHERE matchup_id = %s", (matchup_id,))
    result = cur.fetchone()
    if not result:
        raise HTTPException(status_code=404, detail="Matchup not yet scored.")

    stub_report = (
        f"📊 **AI Post-Game Report** *(AI coach coming soon)*\n\n"
        f"• Team 1 returned **{result['team1_return']:.2f}%** vs Team 2's **{result['team2_return']:.2f}%** this week.\n"
        f"• The winning team's picks outperformed due to broader market momentum and strong sector rotation.\n"
        f"• *(Full AI analysis powered by Claude will be available once API key is configured.)*"
    )
    cur.execute(
        "INSERT INTO pw_ai_reports (matchup_id, report_text) VALUES (%s, %s) "
        "ON CONFLICT (matchup_id) DO NOTHING",
        (matchup_id, stub_report)
    )
    return {"matchup_id": matchup_id, "report": stub_report, "generated_at": datetime.now().isoformat()}


@app.post("/api/wars/draft", status_code=201)
async def submit_draft_pick(body: DraftPick, db=Depends(get_db)):
    """Submit an async draft pick for a team."""
    cur = db.cursor()
    cur.execute("SELECT id FROM pw_teams WHERE id = %s AND league_id = %s", (body.team_id, body.league_id))
    if not cur.fetchone():
        raise HTTPException(status_code=404, detail="Team not found in this league.")
    symbol = body.symbol.upper().strip()
    cur.execute(
        "INSERT INTO pw_draft_picks (league_id, team_id, symbol, pick_round) "
        "VALUES (%s, %s, %s, %s) RETURNING id",
        (body.league_id, body.team_id, symbol, body.pick_round)
    )
    row = cur.fetchone()
    # Update team stocks
    cur.execute("SELECT stocks FROM pw_teams WHERE id = %s", (body.team_id,))
    existing = json.loads(cur.fetchone()["stocks"])
    if symbol not in existing:
        existing.append(symbol)
        cur.execute("UPDATE pw_teams SET stocks = %s WHERE id = %s", (json.dumps(existing), body.team_id))
    return {"id": row["id"], "symbol": symbol, "team_id": body.team_id, "pick_round": body.pick_round}


# ── Trash talk ────────────────────────────────────────────────────────────────

@app.post("/api/wars/message", status_code=201)
async def post_message(body: MessageCreate, db=Depends(get_db)):
    if not body.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")
    cur = db.cursor()
    cur.execute(
        "INSERT INTO pw_messages (league_id, player_name, message) VALUES (%s, %s, %s) RETURNING id",
        (body.league_id, body.player_name.strip(), body.message.strip())
    )
    row = cur.fetchone()
    return {"id": row["id"], "player_name": body.player_name, "message": body.message}


@app.get("/api/wars/messages/{league_id}")
async def get_messages(league_id: int, db=Depends(get_db)):
    cur = db.cursor()
    cur.execute(
        "SELECT * FROM pw_messages WHERE league_id = %s ORDER BY created_at DESC LIMIT 20",
        (league_id,)
    )
    return [dict(r) for r in cur.fetchall()]


@app.post("/api/wars/message/{message_id}/like")
async def like_message(message_id: int, db=Depends(get_db)):
    cur = db.cursor()
    cur.execute("SELECT id FROM pw_messages WHERE id = %s", (message_id,))
    if not cur.fetchone():
        raise HTTPException(status_code=404, detail="Message not found.")
    cur.execute("UPDATE pw_messages SET likes = likes + 1 WHERE id = %s", (message_id,))
    cur.execute("SELECT likes FROM pw_messages WHERE id = %s", (message_id,))
    likes = cur.fetchone()["likes"]
    return {"id": message_id, "likes": likes}


# ════════════════════════════════════════════════════════════════════
# 🔮 PREDICTIONS
# ════════════════════════════════════════════════════════════════════

@app.post("/api/predictions", status_code=201)
async def create_prediction(body: PredictionCreate, db=Depends(get_db), user=Depends(get_current_user)):
    cur = db.cursor()
    cur.execute(
        "INSERT INTO predictions (user_id, symbol, target_price, resolution_date) "
        "VALUES (%s, %s, %s, %s) RETURNING id",
        (int(user["sub"]), body.symbol.upper(), body.target_price, body.resolution_date)
    )
    row = cur.fetchone()
    return {"id": row["id"], "symbol": body.symbol.upper(),
            "target_price": body.target_price, "resolution_date": body.resolution_date}


@app.get("/api/predictions/feed")
async def predictions_feed(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=50),
    db=Depends(get_db)
):
    offset = (page - 1) * limit
    cur = db.cursor()
    cur.execute(
        "SELECT p.*, u.email FROM predictions p "
        "LEFT JOIN users u ON p.user_id = u.id "
        "ORDER BY p.created_at DESC LIMIT %s OFFSET %s",
        (limit, offset)
    )
    rows = cur.fetchall()
    result = []
    for r in rows:
        d = dict(r)
        d["email"] = d["email"].split("@")[0] + "@***" if d.get("email") else "anonymous"
        result.append(d)
    return result


@app.post("/api/predictions/{prediction_id}/like")
async def like_prediction(prediction_id: int, db=Depends(get_db)):
    cur = db.cursor()
    cur.execute("SELECT id FROM predictions WHERE id = %s", (prediction_id,))
    if not cur.fetchone():
        raise HTTPException(status_code=404, detail="Prediction not found.")
    cur.execute("UPDATE predictions SET likes = likes + 1 WHERE id = %s", (prediction_id,))
    cur.execute("SELECT likes FROM predictions WHERE id = %s", (prediction_id,))
    return {"id": prediction_id, "likes": cur.fetchone()["likes"]}


# ════════════════════════════════════════════════════════════════════
# 👤 USER PROFILE
# ════════════════════════════════════════════════════════════════════

@app.get("/api/users/{user_id}/profile")
async def get_user_profile(user_id: int, db=Depends(get_db)):
    cur = db.cursor()
    cur.execute("SELECT id, email, is_pro, created_at FROM users WHERE id = %s", (user_id,))
    user = cur.fetchone()
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")

    # Compute W/L stats across all leagues
    cur.execute(
        "SELECT COUNT(*) AS total FROM pw_results r "
        "JOIN pw_matchups m ON r.matchup_id = m.id "
        "JOIN pw_teams t ON (m.team1_id = t.id OR m.team2_id = t.id) "
        "WHERE t.user_id = %s",
        (user_id,)
    )
    total_matchups = cur.fetchone()["total"]

    cur.execute(
        "SELECT COUNT(*) AS wins FROM pw_results r "
        "JOIN pw_teams t ON r.winner_id = t.id "
        "WHERE t.user_id = %s",
        (user_id,)
    )
    wins = cur.fetchone()["wins"]

    # Predictions accuracy
    cur.execute("SELECT COUNT(*) AS total FROM predictions WHERE user_id = %s", (user_id,))
    total_preds = cur.fetchone()["total"]
    cur.execute(
        "SELECT COUNT(*) AS correct FROM predictions WHERE user_id = %s AND outcome = 'correct'",
        (user_id,)
    )
    correct_preds = cur.fetchone()["correct"]

    # Badges
    cur.execute("SELECT badge_type, earned_at FROM badges WHERE user_id = %s ORDER BY earned_at DESC", (user_id,))
    badges = [dict(b) for b in cur.fetchall()]

    return {
        "id": user["id"],
        "email": user["email"],
        "is_pro": user["is_pro"],
        "member_since": str(user["created_at"]),
        "stats": {
            "total_matchups": total_matchups,
            "wins": wins,
            "losses": total_matchups - wins,
            "win_rate": f"{round(wins / total_matchups * 100)}%" if total_matchups else "N/A",
            "predictions_made": total_preds,
            "prediction_accuracy": f"{round(correct_preds / total_preds * 100)}%" if total_preds else "N/A",
        },
        "badges": badges,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
