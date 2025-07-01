from crewai.tools import BaseTool
import requests
import pandas as pd
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class DataCollectionTool(BaseTool):
    name: str = "data_collection_tool"
    description: str = (
        "Fetches the latest stock data for AAPL from AlphaVantage and appends new rows to the CSV file "
        "without overwriting any sentiment columns. Ensures 'Date' column is preserved in m/d/yyyy format."
    )

    def _run(self, **kwargs) -> str:
        try:
            api_key = os.getenv("ALPHAVANTAGE_API_KEY")
            if not api_key:
                return "❌ ALPHAVANTAGE_API_KEY is missing from environment variables."

            url = (
                "https://www.alphavantage.co/query?"
                "function=TIME_SERIES_DAILY&symbol=AAPL&apikey=" + api_key
            )
            response = requests.get(url)
            data = response.json()

            if "Time Series (Daily)" not in data:
                return f"❌ Failed to retrieve data: {data.get('Note') or data.get('Error Message') or 'Unknown error'}"

            timeseries = data["Time Series (Daily)"]
            df_new = pd.DataFrame.from_dict(timeseries, orient="index")
            df_new.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            df_new.index = pd.to_datetime(df_new.index)
            df_new = df_new.sort_index()
            df_new[['Open', 'High', 'Low', 'Close']] = df_new[['Open', 'High', 'Low', 'Close']].astype(float)
            df_new['Volume'] = df_new['Volume'].astype(float)
            df_new.index.name = "Date"
            df_new = df_new.reset_index()

            csv_path = Path(__file__).resolve().parent.parent / "data" / "aapl_raw.csv"
            csv_path.parent.mkdir(parents=True, exist_ok=True)

            if csv_path.exists():
                df_old = pd.read_csv(csv_path)
                # So sánh theo định dạng chuẩn hóa yyyy-mm-dd để chính xác
                old_dates = set(pd.to_datetime(df_old['Date'], errors='coerce').dt.strftime('%Y-%m-%d'))
            else:
                df_old = pd.DataFrame()
                old_dates = set()

            # So sánh ngày mới với ngày đã có
            df_new['Date_str'] = df_new['Date'].dt.strftime('%Y-%m-%d')
            df_add = df_new[~df_new['Date_str'].isin(old_dates)].copy()
            df_add.drop(columns=['Date_str'], inplace=True)

            if df_add.empty:
                return "✅ No new stock data to append. CSV already up to date."

            # Thêm cột sentiment nếu cần
            sentiment_cols = [col for col in df_old.columns if col.startswith("sentiment")]
            for col in sentiment_cols:
                df_add[col] = None

            # Giữ thứ tự cột nếu có
            if not df_old.empty:
                df_add = df_add.reindex(columns=df_old.columns)
                df_combined = pd.concat([df_old, df_add], ignore_index=True)
            else:
                df_combined = df_add

            # ✅ Format Date về m/d/yyyy (không có số 0 ở đầu)
            df_combined['Date'] = pd.to_datetime(df_combined['Date'], errors='coerce').dt.strftime('%#m/%#d/%Y')

            df_combined.to_csv(csv_path, index=False)

            log = [
                f"✅ Appended {len(df_add)} new row(s) to stock CSV.",
                f"📁 File path: {csv_path.resolve()}",
                f"🧮 Previous rows: {len(df_old)}",
                f"📥 New rows: {len(df_add)}",
                f"📊 Final total rows: {len(df_combined)}",
                f"🕓 Latest appended date: {df_combined['Date'].max()}"
            ]
            return "\n".join(log)

        except Exception as e:
            return f"❌ An unexpected error occurred: {str(e)}"
