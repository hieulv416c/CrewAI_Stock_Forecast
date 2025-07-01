from crewai.tools import BaseTool
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import ta
import os
from pathlib import Path

class Feature_Tools(BaseTool):
    name: str = "data_feature_tool"
    description: str = "Data preprocessing and create more features"

    def _run(self, csv_path: str = "") -> str:
        try:
            csv_path = Path(__file__).resolve().parent.parent / "data" / "aapl_raw.csv"
            csv_path.parent.mkdir(parents=True, exist_ok=True)

            df = pd.read_csv(csv_path)
            df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume',
                     'sentiment_negative', 'sentiment_neutral',
                     'sentiment_positive', 'sentiment_score']].dropna().reset_index(drop=True)

            df['SMA_10'] = df['Close'].rolling(window=10).mean()
            df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
            df['RSI_14'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
            macd = ta.trend.MACD(df['Close'])
            df['MACD'] = macd.macd()
            df = df.dropna().reset_index(drop=True)

            FEATURES = ['Open', 'High', 'Low', 'Close', 'Volume',
                        'sentiment_negative', 'sentiment_neutral',
                        'sentiment_positive', 'sentiment_score',
                        'SMA_10', 'EMA_10', 'RSI_14', 'MACD']
            scalers = {}
            for f in FEATURES:
                scaler = MinMaxScaler()
                df[f + '_scaled'] = scaler.fit_transform(df[[f]])
                scalers[f] = scaler

            scaler_y = MinMaxScaler()
            df['Close_scaled'] = scaler_y.fit_transform(df[['Close']])
            scalers['y'] = scaler_y

            WINDOW_SIZE = 60
            X, y, date_y = [], [], []
            vals = df[[f + '_scaled' for f in FEATURES]].values
            targets = df['Close_scaled'].values
            dates = df['Date'].values

            for i in range(len(df) - WINDOW_SIZE):
                X.append(vals[i:i+WINDOW_SIZE, :])
                y.append(targets[i+WINDOW_SIZE])
                date_y.append(dates[i+WINDOW_SIZE])

            # ➕ Dự đoán cho ngày mai
            X_today = vals[-WINDOW_SIZE:, :]  # shape (60, F)

            X = np.array(X)
            y = np.array(y)
            date_y = np.array(date_y)

            output_dir = Path(__file__).resolve().parent.parent / "Input"
            output_dir.mkdir(parents=True, exist_ok=True)

            np.save(output_dir / 'X_lstm.npy', X)
            np.save(output_dir / 'y_lstm.npy', y)
            np.save(output_dir / 'date_y_lstm.npy', date_y)
            joblib.dump(scalers, output_dir / 'feature_scalers.pkl')
            np.save(output_dir / 'X_today.npy', X_today)  # ➕ dữ liệu hôm nay

            return (
                f"[✅] Đã xử lý feature thành công!\n"
                f"Số sample: {X.shape[0]} | window: {X.shape[1]} | feature: {X.shape[2]}\n"
                f"Đã sinh thêm X_today.npy để dự đoán cho ngày mai."
            )
        except Exception as e:
            return f"[❌] Lỗi khi xử lý feature: {e}"
# Debug run
if __name__ == "__main__":
    tool = Feature_Tools()
    print(tool._run())
