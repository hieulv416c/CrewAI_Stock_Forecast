import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import joblib
import os
from pathlib import Path

# === Đường dẫn cố định theo đúng cấu trúc project của bạn ===
base_dir = Path(__file__).resolve().parents[1]  # => src/tsf_aapl/
input_dir = base_dir / "Input"
models_dir = base_dir / "models"
models_dir.mkdir(parents=True, exist_ok=True)

# === Load dữ liệu đã xử lý ===
X = np.load(input_dir / "X_lstm.npy")  # shape: [N, T, F]
y = np.load(input_dir / "y_lstm.npy")  # đã chuẩn hóa
scalers = joblib.load(input_dir / "feature_scalers.pkl")  # dùng để inverse_transform nếu cần

# === Mô hình LSTM ===
class StockPriceLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.linear(out[:, -1, :])  # chỉ lấy timestep cuối

# === Khởi tạo ===
input_size = X.shape[2]
model = StockPriceLSTM(input_size=input_size)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# === DataLoader ===
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)  # [N, 1]
loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=32, shuffle=True)

# === Huấn luyện ===
EPOCHS = 20
print("=== Đang huấn luyện mô hình ban đầu ===")
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for xb, yb in loader:
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    avg_loss = running_loss / len(loader)
    print(f"[Epoch {epoch+1}/{EPOCHS}] Loss: {avg_loss:.6f}")

# === Lưu model và optimizer ===
torch.save({
    "model_state": model.state_dict(),
    "optimizer_state": optimizer.state_dict()
}, models_dir / "model.pth")

print(f"\n Huấn luyện hoàn tất. Mô hình đã lưu tại {models_dir / 'model.pth'}")
