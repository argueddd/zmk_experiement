import torch
import numpy as np
from torch.utils.data import DataLoader

from under_water_signal_recognize.src.data.dataset import SignalDataset
from under_water_signal_recognize.src.data.preprocess import load_mat_file_into_numpy
from under_water_signal_recognize.src.models.conv1d_classifier import Conv1DRowWiseClassifier
from under_water_signal_recognize.src.utils.mr_loss import batch_mr_loss

# ==== 参数配置 ====
BATCH_SIZE = 16
LR = 1e-3
EPOCHS = 50
LAMBDA = 0.5
SIGMA = 0.0001
PATIENCE = 5

# ==== 加载数据 ====
features, labels = load_mat_file_into_numpy('../data/Wmel_Feature.mat', '../data/Label.mat')
features = features.astype(np.float32)
print("train feature shape:", features.shape)

labels = labels.squeeze().astype(np.int64) - 1

train_loader = DataLoader(SignalDataset(features, labels), batch_size=BATCH_SIZE, shuffle=True)

# ==== 初始化模型 ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Conv1DRowWiseClassifier(num_rows=features.shape[1], num_classes=4).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = torch.nn.CrossEntropyLoss()

# ==== Early Stopping ====
best_loss = float('inf')
best_model_state = None
epochs_no_improve = 0

# ==== 训练循环 ====
for epoch in range(EPOCHS):
    model.train()
    total_loss_val = 0

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        output = model(X_batch)
        ce_loss = criterion(output, y_batch)
        mr_loss = batch_mr_loss(X_batch, sigma=SIGMA)
        total_loss = ce_loss + LAMBDA * mr_loss
        total_loss.backward()
        optimizer.step()
        total_loss_val += total_loss.item()

    avg_loss = total_loss_val / len(train_loader)
    print(f"Epoch {epoch + 1}/{EPOCHS} - Loss: {avg_loss:.4f}")

    # ==== Early Stopping 检查 ====
    if avg_loss < best_loss - 1e-4:
        best_loss = avg_loss
        best_model_state = model.state_dict()
        epochs_no_improve = 0
        print("New best model found and saved.")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            print(f"Early stopping triggered at epoch {epoch + 1}.")
            break

# ==== 保存模型 ====
if best_model_state is not None:
    torch.save(best_model_state, "conv1d_classifier_mr.pth")
    print("Model saved to conv1d_classifier_mr.pth")
else:
    torch.save(model.state_dict(), "conv1d_classifier_last.pth")
    print("Model saved to conv1d_classifier_last.pth (no improvement)")
