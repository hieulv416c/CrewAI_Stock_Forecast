import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Đọc file dữ liệu các bài báo
df = pd.read_csv('apple_news_mediastack_6months.csv')

# Chuyển date về đúng format YYYY-MM-DD
df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')

# Tạo đối tượng vader
analyzer = SentimentIntensityAnalyzer()

# Kết quả lưu ở đây
sentiment_by_day = []

# Group các bài báo theo ngày
for date, group in df.groupby('date'):
    neg, neu, pos, comp = [], [], [], []
    for _, row in group.iterrows():
        # Lấy text để phân tích, ưu tiên title + description
        text = (str(row['title']) or '') + '. ' + (str(row['description']) or '')
        if text.strip():
            scores = analyzer.polarity_scores(text)
            neg.append(scores['neg'])
            neu.append(scores['neu'])
            pos.append(scores['pos'])
            comp.append(scores['compound'])
    if comp:
        sentiment_by_day.append({
            'date': date,
            'sentiment_negative': sum(neg) / len(neg),
            'sentiment_neutral': sum(neu) / len(neu),
            'sentiment_positive': sum(pos) / len(pos),
            'sentiment_compound': sum(comp) / len(comp),
            'n_articles': len(comp)
        })
    else:
        sentiment_by_day.append({
            'date': date,
            'sentiment_negative': 0,
            'sentiment_neutral': 1,
            'sentiment_positive': 0,
            'sentiment_compound': 0,
            'n_articles': 0
        })

# Lưu ra file mới, sort ngày tăng dần
df_sentiment = pd.DataFrame(sentiment_by_day)
df_sentiment = df_sentiment.sort_values('date').reset_index(drop=True)
df_sentiment.to_csv('apple_news_sentiment_by_day.csv', index=False)
print(df_sentiment.head())
