# src/tsf_aapl/tools/predict_today_tool.py

from crewai.tools import BaseTool
import numpy as np
import torch
import joblib
import os
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

class StockPriceLSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = torch.nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.linear(out[:, -1, :])

class PredictTodayTool(BaseTool):
    name: str = "predict_today_tool"
    description: str = (
        "Predict the AAPL stock price for today (next available date) using most recent data. "
        "Uses LSTM model and inverse scaling to get the actual predicted price."
    )

    def _run(self, **kwargs) -> str:
        root = Path(__file__).resolve().parent.parent
        input_dir = root / "Input"
        model_path = root / "models" / "model.pth"
        log_path = root / "logs" / "predict_log.csv"

        try:
            X_path = input_dir / "X_today.npy"
            scaler_path = input_dir / "feature_scalers.pkl"

            if not X_path.exists():
                return "❌ Missing X_today.npy. Hãy chắc chắn đã chạy Feature_Tools sau khi thêm dữ liệu mới nhất."

            X_today = np.load(X_path)
            X_tensor = torch.tensor(X_today[np.newaxis, :, :], dtype=torch.float32)

            input_size = X_today.shape[1]
            model = StockPriceLSTM(input_size=input_size)
            checkpoint = torch.load(model_path)
            model.load_state_dict(checkpoint["model_state"])
            model.eval()

            with torch.no_grad():
                pred_scaled = model(X_tensor).item()

            scalers = joblib.load(scaler_path)
            pred_real = scalers["y"].inverse_transform([[pred_scaled]])[0][0] if "y" in scalers else None

            # Use date format compatible with date_y_lstm.npy (e.g., "6/10/2025")
            predict_date = datetime.today().strftime("%-m/%-d/%Y") if os.name != "nt" else datetime.today().strftime("%#m/%#d/%Y")

            # Ensure log folder exists
            os.makedirs(log_path.parent, exist_ok=True)

            # Check if today's prediction already exists in the log
            if log_path.exists():
                df_log = pd.read_csv(log_path, header=None)
                if not df_log[df_log[0] == predict_date].empty:
                    return f"[ℹ️] Prediction for {predict_date} already exists in log. Skipping re-log."

            # Write new log entry
            with open(log_path, "a") as logf:
                logf.write(f"{predict_date},{pred_scaled:.5f},{pred_real if pred_real else 'N/A'}\n")

            return "\n".join([
                f"[📈] Dự đoán giá Close AAPL cho {predict_date}: {pred_real:.2f} USD" if pred_real else f"Scaled: {pred_scaled:.4f}",
                f"Model: {model_path.name}",
                f"Log: {log_path.name}"
            ])

        except Exception as e:
            return f"[❌] Error during prediction: {e}"

# Debug tool nếu gọi trực tiếp
if __name__ == "__main__":
    tool = PredictTodayTool()
    output = tool._run()
    print("[TOOL RESULT]")
    print(output)
