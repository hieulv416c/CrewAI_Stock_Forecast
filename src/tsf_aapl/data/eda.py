# import pandas as pd

# # Đọc file sentiment đã chuẩn bị
# df_sent = pd.read_csv('apple_sentiment_final_for_lstm.csv')
# df_sent['date'] = pd.to_datetime(df_sent['date'])
# sentiment_days = set(df_sent['date'].dt.strftime('%Y-%m-%d'))

# # Đọc file giá cổ phiếu (cột Date hoặc date, kiểu yyyy-mm-dd)
# df_price = pd.read_csv('aapl_raw.csv')
# # Đổi tên cột về đúng tên nếu khác
# df_price['Date'] = pd.to_datetime(df_price['Date'])
# price_days = set(df_price['Date'].dt.strftime('%Y-%m-%d'))

# # Giới hạn khoảng thời gian theo file sentiment
# min_date = '1/01/2024'
# min_date = pd.to_datetime(min_date)
# max_date = df_sent['date'].max()
# trading_days = {d for d in price_days if min_date <= pd.to_datetime(d) <= max_date}

# # Ngày giao dịch bị thiếu sentiment
# missing_days = sorted(trading_days - sentiment_days)

# print(f"Số ngày giao dịch thiếu sentiment: {len(missing_days)}")
# print("Các ngày bị thiếu:", missing_days)
# df_missing_days = pd.DataFrame(missing_days)
# df_missing_days.to_csv('missingdays.csv', index = False)


# import pandas as pd

# # Đọc dữ liệu
# price_df = pd.read_csv("aapl_raw.csv")
# sentiment_df = pd.read_csv("apple_sentiment_final_for_lstm.csv")

# # Chuẩn hóa tên cột ngày (giả sử 2 file là 'Date' và 'date')
# price_df['Date'] = pd.to_datetime(price_df['Date'])
# sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])

# # Merge theo ngày
# merged = pd.merge(price_df, sentiment_df, left_on='Date', right_on='date', how='left')

# # Lọc từ ngày 01/01/2021 trở đi
# merged = merged[merged['Date'] >= pd.to_datetime("2021-01-01")]

# # Bỏ cột 'date' dư thừa nếu muốn
# merged = merged.drop(columns=['date'])

# # Xuất ra file mới
# merged.to_csv("aapl_merged_with_sentiment_2021.csv", index=False)
# print("Đã lưu file aapl_merged_with_sentiment_2021.csv")


# import pandas as pd
# import requests
# from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
# from datetime import datetime, timedelta

# API_KEY = '65ff844dce314c4fb69af981ee1813c4'
# CSV_PATH = 'src/tsf_aapl/data/aapl_merged_with_sentiment_2021.csv'
# NEWS_KEYWORDS = 'AAPL OR Apple stock'
# MAX_ARTICLES_PER_DAY = 50

# def get_news_for_time_range(day, api_key, keywords, max_articles=50):
#     # Lấy 10h sáng ngày hôm trước và 10h sáng hôm nay
#     day_dt = datetime.strptime(day, '%Y-%m-%d')
#     from_dt = day_dt - timedelta(days=1)
#     from_dt = from_dt.replace(hour=10, minute=0, second=0)
#     to_dt = day_dt.replace(hour=10, minute=0, second=0)
#     from_str = from_dt.strftime('%Y-%m-%dT%H:%M:%S')
#     to_str = to_dt.strftime('%Y-%m-%dT%H:%M:%S')
#     url = 'https://newsapi.org/v2/everything'
#     params = {
#         'q': keywords,
#         'from': from_str,
#         'to': to_str,
#         'language': 'en',
#         'sortBy': 'publishedAt',
#         'pageSize': max_articles,
#         'apiKey': api_key
#     }
#     r = requests.get(url, params=params)
#     data = r.json()
#     return data.get('articles', [])

# def sentiment_scores(articles):
#     analyzer = SentimentIntensityAnalyzer()
#     neg, neu, pos, comp = [], [], [], []
#     for article in articles:
#         text = (article.get('title') or '') + '. ' + \
#                (article.get('description') or '') + '. ' + \
#                (article.get('content') or '')
#         if text.strip():
#             scores = analyzer.polarity_scores(text)
#             neg.append(scores['neg'])
#             neu.append(scores['neu'])
#             pos.append(scores['pos'])
#             comp.append(scores['compound'])
#     if comp:
#         return (
#             sum(neg) / len(neg),
#             sum(neu) / len(neu),
#             sum(pos) / len(pos),
#             sum(comp) / len(comp)
#         )
#     else:
#         return (0.0, 1.0, 0.0, 0.0)

# df = pd.read_csv(CSV_PATH)
# df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')

# for col in ['sentiment_negative', 'sentiment_neutral', 'sentiment_positive', 'sentiment_score']:
#     if col not in df.columns:
#         df[col] = None

# for idx, row in df.iterrows():
#     if pd.isna(row['sentiment_score']) or pd.isna(row['sentiment_negative']):
#         day = row['Date']
#         print(f"Crawling news for {day} (from 10AM previous day to 10AM that day)...")
#         articles = get_news_for_time_range(day, API_KEY, NEWS_KEYWORDS, MAX_ARTICLES_PER_DAY)
#         neg, neu, pos, comp = sentiment_scores(articles)
#         df.at[idx, 'sentiment_negative'] = neg
#         df.at[idx, 'sentiment_neutral'] = neu
#         df.at[idx, 'sentiment_positive'] = pos
#         df.at[idx, 'sentiment_score'] = comp
#         print(f"Done {day}: {len(articles)} articles, Sentiment: {neg:.3f}/{neu:.3f}/{pos:.3f}/{comp:.3f}")

# df.to_csv(CSV_PATH, index=False)
# print("Đã cập nhật xong file!")
import pandas as pd

df = pd.read_csv('src/tsf_aapl/data/aapl_raw.csv')
df.rename(columns={'date': 'Date'}, inplace=True)
df.to_csv('src/tsf_aapl/data/aapl_raw.csv', index=False)