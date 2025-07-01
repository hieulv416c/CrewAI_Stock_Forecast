from crewai.tools import BaseTool
import numpy as np
import torch
import torch.nn as nn
import joblib
from pathlib import Path
import os
from datetime import datetime
from scipy.stats import entropy
from sklearn.random_projection import SparseRandomProjection

# === Mô hình LSTM ===
class StockPriceLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.linear(out[:, -1, :])

# === Tool cập nhật mô hình ===
class PredictUpdateTool(BaseTool):
    name: str = "predict_update_tool"
    description: str = (
        "Detect drift using KL divergence, apply ELM if drift exists, and fine-tune the model with the latest sample."
    )

    def _run(self, **kwargs) -> str:
        try:
            root = Path(__file__).resolve().parent.parent
            X_path = root / "Input" / "X_lstm.npy"
            y_path = root / "Input" / "y_lstm.npy"
            date_path = root / "Input" / "date_y_lstm.npy"
            scaler_path = root / "Input" / "feature_scalers.pkl"
            model_path = root / "models" / "model.pth"
            log_path = root / "logs" / "lstm_online_log.csv"

            if not all(p.exists() for p in [X_path, y_path, date_path, model_path, scaler_path]):
                return "[❌] Missing one or more required input files."

            # Load data
            X_all = np.load(X_path)
            y_all = np.load(y_path)
            date_all = np.load(date_path)
            scalers = joblib.load(scaler_path)

            X_yesterday = X_all[-1]
            y_true_scaled = y_all[-1]
            date_yesterday = str(date_all[-1])

            # Drift detection with KL Divergence
            last_window = X_all[-1]
            prev_window = X_all[-2]
            kl_total = 0
            for i in range(last_window.shape[1]):
                p = prev_window[:, i] + 1e-10
                q = last_window[:, i] + 1e-10
                p /= p.sum()
                q /= q.sum()
                kl_total += entropy(p, q)
            kl_avg = kl_total / last_window.shape[1]
            drift_detected = kl_avg > 0.1

            print(f"[🔍] Drift detected: {drift_detected} | KL Divergence: {kl_avg:.5f}")

            # Apply ELM projection if drift
            if drift_detected:
                flat_input = X_yesterday.reshape(-1, X_yesterday.shape[1])
                projector = SparseRandomProjection(n_components='auto', eps=0.5)
                transformed = projector.fit_transform(flat_input).reshape(1, 60, -1)
                X_tensor = torch.tensor(transformed, dtype=torch.float32)
            else:
                X_tensor = torch.tensor(X_yesterday[np.newaxis, :, :], dtype=torch.float32)

            # Load model
            model = StockPriceLSTM(input_size=X_tensor.shape[2])
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
            checkpoint = torch.load(model_path)
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])

            # Fine-tune
            model.train()
            y_tensor = torch.tensor([[y_true_scaled]], dtype=torch.float32)
            optimizer.zero_grad()
            output = model(X_tensor)
            loss = nn.MSELoss()(output, y_tensor)
            loss.backward()
            optimizer.step()

            # Save updated model
            torch.save({
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict()
            }, model_path)

            # Log
            os.makedirs(log_path.parent, exist_ok=True)
            y_true_real = scalers["y"].inverse_transform([[y_true_scaled]])[0][0]
            y_pred_real = scalers["y"].inverse_transform([[output.item()]])[0][0]
            error = abs(y_true_real - y_pred_real)
            with open(log_path, "a") as f:
                f.write(f"{date_yesterday},{y_true_real:.4f},{y_pred_real:.4f},{error:.6f},{kl_avg:.5f},{'drift' if drift_detected else 'stable'}\n")

            return (
                f"Final Answer: Model updated using data from {date_yesterday}.\n"
                f"True: {y_true_real:.2f} | Predicted: {y_pred_real:.2f} | Error: {error:.6f}\n"
                f"Drift: {'Yes' if drift_detected else 'No'} | KL Divergence: {kl_avg:.5f}\n"
                f"Log saved to: logs/lstm_online_log.csv"
                )

        except Exception as e:
            return f"[❌] Error in PredictUpdateTool: {e}"

# Debug run
if __name__ == "__main__":
    tool = PredictUpdateTool()
    print(tool._run())
