from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np

analyzer = SentimentIntensityAnalyzer()

def analyze_sentiment(text):
    if not text:
        return {"compound": 0.0, "sentiment": "neutral"}
    
    score = analyzer.polarity_scores(text)
    compound = score["compound"]
    
    if compound >= 0.05:
        sentiment = "positive"
    elif compound <= -0.05:
        sentiment = "negative"
    else:
        sentiment = "neutral"
    
    return {
        "compound": compound,
        "sentiment": sentiment,
        "positive": score["pos"],
        "negative": score["neg"],
        "neutral": score["neu"]
    }

def analyze_articles(news_data):
    if not news_data or 'articles' not in news_data:
        return {
            "error": "Invalid news data format",
            "articles": [],
            "overall_sentiment": "neutral"
        }
    
    results = []
    for article in news_data['articles']:
        title = article.get("title", "")
        description = article.get("description", "") or ""
        content = article.get("content", "") or ""
        
        # Combine title and description for better analysis of the article
        full_text = f"{title}. {description}"
        sentiment = analyze_sentiment(full_text)
        
        results.append({
            "title": title,
            "url": article.get("url", "#"),
            "source": article.get("source", {}).get("name", "Unknown"),
            "publishedAt": article.get("publishedAt", ""),
            "sentiment": sentiment,
            "image": article.get("urlToImage", "")
        })
    
    # Calculate overall sentiment
    compounds = [r["sentiment"]["compound"] for r in results]
    avg_sentiment = np.mean(compounds) if compounds else 0
    
    return {
        "symbol": news_data.get('symbol', ''),
        "articles": results,
        "overall_sentiment": analyze_sentiment_from_score(avg_sentiment),
        "article_count": len(results)
    }

def analyze_sentiment_from_score(score):
    if score >= 0.05:
        return "positive"
    elif score <= -0.05:
        return "negative"
    else:
        return "neutral"