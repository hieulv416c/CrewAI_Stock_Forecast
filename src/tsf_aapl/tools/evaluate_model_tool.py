# src/tsf_aapl/tools/evaluate_tool.py
from crewai.tools import BaseTool
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import datetime

class EvaluateModelTool(BaseTool):
    name: str = "evaluate_model_tool"
    description: str = "Evaluate prediction accuracy over the last 5 and 30 days using logs."

    def _run(self, **kwargs) -> str:
        try:
            ROOT = Path(__file__).resolve().parent.parent
            predict_log_path = ROOT / "logs" / "predict_log.csv"
            raw_csv_path = ROOT / "data" / "aapl_raw.csv"
            short_log_path = ROOT / "logs" / "lstm_eval_short_log.csv"
            long_log_path = ROOT / "logs" / "lstm_eval_long_log.csv"

            if not predict_log_path.exists() or not raw_csv_path.exists():
                return "[❌] Thiếu file predict_log.csv hoặc aapl_raw.csv"

            df_log = pd.read_csv(predict_log_path)
            df_log.columns = ["Date", "Pred_Scaled", "Pred_Real"]

            df_price = pd.read_csv(raw_csv_path)
            df_price["Date"] = pd.to_datetime(df_price["Date"]).dt.date
            df_log["Date"] = pd.to_datetime(df_log["Date"]).dt.date

            merged = df_log.merge(df_price[["Date", "Close"]], on="Date", how="left").dropna()

            if merged.empty:
                return "[⚠️] Không có dòng nào có đủ dữ liệu thực tế để đánh giá"

            short = merged.tail(5)
            mae_5 = mean_absolute_error(short["Close"], short["Pred_Real"])
            rmse_5 = np.sqrt(mean_squared_error(short["Close"], short["Pred_Real"]))
            mape_5 = np.mean(np.abs((short["Close"] - short["Pred_Real"]) / short["Close"])) * 100

            today_str = datetime.today().strftime("%Y-%m-%d")
            short_log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(short_log_path, "a") as f:
                f.write(f"{today_str},{mae_5:.4f},{rmse_5:.4f},{mape_5:.2f}\n")

            result = [f"[📊] Đánh giá mô hình dựa trên log dự đoán\n",
                      f"[🔎] 5 ngày gần nhất:",
                      f"MAE: {mae_5:.4f} | RMSE: {rmse_5:.4f} | MAPE: {mape_5:.2f}%"]

            if len(merged) >= 30:
                long = merged.tail(30)
                mae_30 = mean_absolute_error(long["Close"], long["Pred_Real"])
                rmse_30 = np.sqrt(mean_squared_error(long["Close"], long["Pred_Real"]))
                mape_30 = np.mean(np.abs((long["Close"] - long["Pred_Real"]) / long["Close"])) * 100

                with open(long_log_path, "a") as f:
                    f.write(f"{today_str},{mae_30:.4f},{rmse_30:.4f},{mape_30:.2f}\n")

                result += [f"\n[📈] 30 ngày gần nhất:",
                           f"MAE: {mae_30:.4f} | RMSE: {rmse_30:.4f} | MAPE: {mape_30:.2f}%"]
            else:
                result.append("\n[⚠️] Không đủ dữ liệu để đánh giá 30 ngày.")

            return "\n".join(result)

        except Exception as e:
            return f"[❌] Lỗi trong EvaluateModelTool: {e}"
