# Quick Deployment Checklist

## 1. Initialize Git
```powershell
cd c:\Users\ninai\stockapp
git init
git add .
git commit -m "Initial commit - Stock Prediction App"
```

## 2. Create GitHub Repo
- Go to https://github.com/new
- Name: `stock-prediction-app`
- Make it **Public**
- Don't initialize with README
- Click **Create**

## 3. Push to GitHub
```powershell
# Replace YOUR_USERNAME with your actual GitHub username
git remote add origin https://github.com/YOUR_USERNAME/stock-prediction-app.git
git branch -M main
git push -u origin main
```

## 4. Deploy to Render.com
1. Go to https://render.com
2. Sign up with GitHub
3. **New +** → **Web Service**
4. Connect your `stock-prediction-app` repository
5. Configure:
   - **Name:** `stock-prediction-app`
   - **Runtime:** `Python 3`
   - **Build:** `pip install -r requirements.txt`
   - **Start:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - **Instance:** **Free**

6. **Environment Variables:**
   - Add: `NEWS_API_KEY` = (your key from https://newsapi.org)
   - Add: `PYTHON_VERSION` = `3.11.0`

7. **Add Disk:**
   - Name: `data-disk`
   - Mount Path: `/opt/render/project/src/data`
   - Size: `1 GB`

8. Click **Create Web Service**

## 5. Wait & Test
- Build takes 5-10 minutes
- Your app will be at: `https://stock-prediction-app-xxxx.onrender.com`
- Test: `https://your-app.onrender.com/docs`

Done! 🎉
