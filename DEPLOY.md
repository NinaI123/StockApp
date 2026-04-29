# Deployment Guide

## What's in the App
- **AI Stock Analyzer** – LSTM predictions, sentiment analysis, technical indicators
- **⚔️ Portfolio Wars** – Fantasy sports-style weekly stock competition (leagues, teams, matchups, live scoring)

---

## 1. Initialize Git
```powershell
cd c:\Users\ninai\stockapp
git init
git add .
git commit -m "Add Portfolio Wars feature"
```

## 2. Create GitHub Repo
- Go to https://github.com/new → Name: `stock-prediction-app` → Public → Create

## 3. Push to GitHub
```powershell
git remote add origin https://github.com/YOUR_USERNAME/stock-prediction-app.git
git branch -M main
git push -u origin main
```

## 4. Deploy to Render.com
1. Go to https://render.com → Sign up with GitHub
2. **New +** → **Web Service** → Connect `stock-prediction-app`
3. Configure:
   - **Runtime:** `Python 3`
   - **Build:** `pip install -r requirements.txt`
   - **Start:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - **Instance:** Free (or Starter for always-on)

4. **Environment Variables:**
   | Key | Value |
   |-----|-------|
   | `NEWS_API_KEY` | Your key from https://newsapi.org |
   | `PYTHON_VERSION` | `3.11.0` |
   | `DATABASE_PATH` | `/opt/render/project/src/data/portfolio.db` |

5. **Add Persistent Disk** *(required — keeps DB alive across deploys)*:
   - Name: `data-disk`
   - Mount Path: `/opt/render/project/src/data`
   - Size: `1 GB`

6. Click **Create Web Service**

## 5. Test After Deploy
- App: `https://your-app.onrender.com`
- API Docs: `https://your-app.onrender.com/docs`
- Portfolio Wars: click **⚔️ Portfolio Wars** in the navbar

---

## Scaling (when you're ready)
```
uvicorn main:app --workers 4 --host 0.0.0.0 --port $PORT
```

## Upgrade Path: Multi-user DB
If you want true multi-user (many concurrent leagues), swap SQLite for Supabase/Neon:
1. Add `DATABASE_URL` env var → e.g. `postgresql://...`
2. Replace `sqlite3` with `asyncpg` in `main.py`
3. Port the `CREATE TABLE` SQL statements (mostly compatible)

Done! 🎉
