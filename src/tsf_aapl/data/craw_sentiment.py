import pandas as pd
import requests
import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

API_KEY = '65ff844dce314c4fb69af981ee1813c4'  # Đổi thành key của bạn nếu cần
url = 'https://newsapi.org/v2/everything'

# 1. Lấy list ngày giao dịch (NYSE)
ticker = yf.Ticker("AAPL")
hist = ticker.history(start="2024-11-28", end="2025-06-05")
trading_days = [d.strftime('%Y-%m-%d') for d in hist.index]

analyzer = SentimentIntensityAnalyzer()
results = []

for day in trading_days:
    from_iso = f"{day}T00:00:00Z"
    to_iso = f"{day}T23:59:59Z"
    params = {
        'q': 'Apple',  # Có thể thử các từ khoá như 'AAPL OR Apple stock' nếu muốn
        'from': from_iso,
        'to': to_iso,
        'language': 'en',
        'sortBy': 'publishedAt',
        'pageSize': 100,         # Tối đa 100 bài/newsapi 1 lần
        'apiKey': API_KEY
    }
    response = requests.get(url, params=params)
    data = response.json()
    articles = data.get('articles', [])
    print(f"Ngày API query: {from_iso} - {to_iso}, trả về {len(articles)} bài báo")

    neg, neu, pos, comp = [], [], [], []
    for article in articles[:50]:   # Tối đa 50 bài/ngày (có thể điều chỉnh)
        text = (article.get('title') or '') + '. ' + (article.get('description') or '') + '. ' + (article.get('content') or '')
        if text.strip():
            scores = analyzer.polarity_scores(text)
            neg.append(scores['neg'])
            neu.append(scores['neu'])
            pos.append(scores['pos'])
            comp.append(scores['compound'])
    if comp:
        result = {
            'date': day,
            'sentiment_negative': sum(neg) / len(neg),
            'sentiment_neutral': sum(neu) / len(neu),
            'sentiment_positive': sum(pos) / len(pos),
            'sentiment_compound': sum(comp) / len(comp),
            'n_articles': len(comp)
        }
    else:
        result = {
            'date': day,
            'sentiment_negative': 0,
            'sentiment_neutral': 1,
            'sentiment_positive': 0,
            'sentiment_compound': 0,
            'n_articles': 0
        }
    results.append(result)

# Lưu kết quả ra file
df = pd.DataFrame(results)
df.to_csv('aapl_sentiment_vader_by_trading_day.csv', index=False)
print(df.head())
