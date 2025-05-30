import numpy as np
import torch
from torch.utils.data import DataLoader

from under_water_signal_recognize.src.conv1d_classifier import Conv1DRowWiseClassifier
from under_water_signal_recognize.src.dataset import SignalDataset
from under_water_signal_recognize.src.utils import load_mat_file_into_numpy

# 加载数据
data_path = '../data/Wmel_Feature.mat'
label_path = '../data/Label.mat'
features, labels = load_mat_file_into_numpy(data_path, label_path)

features = features.astype(np.float32)
labels = labels.squeeze().astype(np.int64) - 1

train_loader = DataLoader(SignalDataset(features, labels), batch_size=16, shuffle=True)

# 模型 & 训练配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Conv1DRowWiseClassifier(num_rows=features.shape[1], num_classes=4).to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Early Stopping 参数
patience = 3
delta = 1e-4
best_loss = float('inf')
epochs_no_improve = 0
best_model_state = None

# 训练循环
max_epochs = 50
for epoch in range(max_epochs):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1} | Loss: {avg_loss:.4f}")

    # Early Stopping 检查
    if avg_loss < best_loss - delta:
        best_loss = avg_loss
        epochs_no_improve = 0
        best_model_state = model.state_dict()
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch + 1}. Best loss: {best_loss:.4f}")
            break

# 保存最佳模型
if best_model_state is not None:
    torch.save(best_model_state, "conv1d_classifier.pth")
    print("Best model saved as conv1d_classifier.pth")
else:
    torch.save(model.state_dict(), "conv1d_classifier.pth")
    print("Model saved as conv1d_classifier.pth (no improvement)")
