import requests
from datetime import datetime, timedelta

NEWS_API_KEY = "34d22fcdceaf4990879b7a5600a3eba1"

def fetch_news(query, page_size=10):
    """Fetch news articles for a given stock symbol"""
    # Get news - do 30 if you want to get news from the last month
    from_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    
    url = f"https://newsapi.org/v2/everything?q={query}&apiKey={NEWS_API_KEY}" \
          f"&language=en&pageSize={page_size}&from={from_date}&sortBy=publishedAt"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        articles = response.json().get("articles", [])
        
        # Filter out articles without content
        filtered_articles = [article for article in articles if article.get("title") and article.get("url")]
        
        return {
            "symbol": query,
            "articles": filtered_articles,
            "count": len(filtered_articles),
            "last_updated": datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"Error fetching news: {e}")
        return {
            "symbol": query,
            "articles": [],
            "count": 0,
            "last_updated": datetime.now().isoformat(),
            "error": str(e)
        }