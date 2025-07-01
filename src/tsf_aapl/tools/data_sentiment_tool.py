from crewai.tools import BaseTool
import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

load_dotenv()

class SentimentAnalysisTool(BaseTool):
    name: str = "SentimentAnalysisTool"
    description: str = (
        "Performs sentiment analysis on news articles related to AAPL using NewsAPI, "
        "and updates the CSV file with four sentiment columns: sentiment_positive, "
        "sentiment_negative, sentiment_neutral, and sentiment_score. "
        "Only processes rows that do not already have a sentiment_score."
    )

    def _run(self, limit: int = 5, **kwargs) -> str:
        try:
            api_key = os.getenv("NEWSAPI_KEY")
            if not api_key:
                return "❌ NEWSAPI_KEY is missing from environment variables."

            csv_path = Path(__file__).resolve().parent.parent / "data" / "aapl_raw.csv"
            if not csv_path.exists():
                return f"❌ CSV file not found at: {csv_path.resolve()}"

            df = pd.read_csv(csv_path)
            if 'Date' not in df.columns:
                return "❌ CSV does not contain 'Date' column."

            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df = df.dropna(subset=['Date']).sort_values('Date')

            if 'sentiment_score' not in df.columns:
                df['sentiment_score'] = pd.NA

            to_update = df[df['sentiment_score'].isna()].copy()
            if to_update.empty:
                return "✅ Final Answer: No rows required sentiment update. All rows in aapl_raw.csv already have sentiment scores."

            analyzer = SentimentIntensityAnalyzer()
            updates = 0
            updated_dates = []

            for idx, row in to_update.head(limit).iterrows():
                date_str = row['Date'].strftime('%m/%d/%Y')
                from_time = (row['Date'] - timedelta(days=1)).replace(hour=10)
                to_time = row['Date'].replace(hour=10)

                print(f"[{date_str}] ▶ Gửi yêu cầu NewsAPI từ {from_time} đến {to_time}")
                response = requests.get(
                    "https://newsapi.org/v2/everything",
                    params={
                        'q': 'AAPL',
                        'from': from_time.isoformat(),
                        'to': to_time.isoformat(),
                        'sortBy': 'relevancy',
                        'language': 'en',
                        'pageSize': 50,
                        'apiKey': api_key
                    }
                )
                data = response.json()
                articles = data.get('articles', [])
                print(f"[{date_str}]  Số bài báo: {len(articles)}")

                # === Ghi log số lượng bài báo ===
                log_dir = Path(__file__).resolve().parent.parent / "logs"
                log_dir.mkdir(parents=True, exist_ok=True)
                log_file = log_dir / "log_paper.csv"

                log_entry = pd.DataFrame([{
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'date': date_str,
                    'article_count': len(articles)
                }])
                if log_file.exists():
                    log_entry.to_csv(log_file, mode='a', header=False, index=False)
                else:
                    log_entry.to_csv(log_file, mode='w', header=True, index=False)

                # === Phân tích sentiment ===
                if not articles:
                    sentiment = {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
                else:
                    texts = [
                        (str(a.get('title') or '') + " " + str(a.get('description') or '')).strip()
                        for a in articles
                    ]
                    scores = [analyzer.polarity_scores(text) for text in texts]
                    sentiment = {
                        'neg': round(sum(s['neg'] for s in scores) / len(scores), 4),
                        'neu': round(sum(s['neu'] for s in scores) / len(scores), 4),
                        'pos': round(sum(s['pos'] for s in scores) / len(scores), 4),
                        'compound': round(sum(s['compound'] for s in scores) / len(scores), 4)
                    }

                df_idx = df.index[df['Date'] == row['Date']][0]
                df.at[df_idx, 'sentiment_negative'] = sentiment['neg']
                df.at[df_idx, 'sentiment_neutral'] = sentiment['neu']
                df.at[df_idx, 'sentiment_positive'] = sentiment['pos']
                df.at[df_idx, 'sentiment_score'] = sentiment['compound']

                updates += 1
                updated_dates.append(date_str)
                print(f"[{date_str}]  ✅ Đã cập nhật sentiment.")

            # Format lại ngày
            df['Date'] = df['Date'].dt.strftime('%#m/%#d/%Y' if os.name == 'nt' else '%-m/%-d/%Y')
            df.to_csv(csv_path, index=False, float_format='%.4f')

            return (
                f"✅ Updated sentiment for {updates} date(s):\n"
                + ", ".join(updated_dates)
                + f"\n📁 File path: {csv_path.resolve()}\n"
                + f"📝 Log đã ghi vào: {log_file.resolve()}"
            )

        except Exception as e:
            return f"❌ An unexpected error occurred: {str(e)}"
