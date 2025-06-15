import time

import torch
import numpy as np
from torch.utils.data import DataLoader

from under_water_signal_recognize.src.data.dataset import SignalDataset
from under_water_signal_recognize.src.models.conv1d_classifier import Conv1DRowWiseClassifier, cumulative_bce_loss, \
    bce_loss
from under_water_signal_recognize.src.utils.mr_loss import batch_mr_loss

# ==== 参数配置 ====
BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 100
ALP = 1
LAMBDA = 0.7
SIGMA = 0.5
PATIENCE = 5

# ==== 加载数据 ====
data = np.load('data/DeepShip/npz/deepship_trained_dataset_window.npz')
features = data['features']
print("eval feature shape:", features.shape)
labels = data['labels']
train_loader = DataLoader(SignalDataset(features, labels), batch_size=BATCH_SIZE, shuffle=True)

# ==== 初始化模型 ====
device = torch.device("mps")
# device = torch.device("mps" if torch.cuda.is_available() else "cpu")
model = Conv1DRowWiseClassifier(num_rows=features.shape[1], num_classes=4).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)


# ==== Early Stopping ====
best_loss = float('inf')
best_model_state = None
epochs_no_improve = 0

# ==== 训练循环 ====
for epoch in range(EPOCHS):
    start_time = time.time()
    model.train()
    total_loss_val = 0

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        # 将原始标签转换为累积概率标签
        batch_size = y_batch.size(0)
        num_classes = 4  # 根据你的模型设置

        # 创建全1累积标签矩阵（初始化为所有类累积概率为1）
        cumulative_labels = torch.ones(batch_size, num_classes, device=device)

        # 按规则生成标签：label=k时，前k个位置设为0
        for i in range(batch_size):
            label = y_batch[i].item()
            # 前label个位置设为0（从第0列到第label-1列）
            if label > 0:
                cumulative_labels[i, :label] = 0.0

        optimizer.zero_grad()
        output = model(X_batch)

        # 使用累积概率损失
        # ce_loss = cumulative_bce_loss(output, cumulative_labels)
        ce_loss = bce_loss(output, y_batch)

        mr_loss = batch_mr_loss(X_batch, sigma=SIGMA)
        total_loss = ALP * ce_loss + LAMBDA * mr_loss
        total_loss.backward()
        optimizer.step()
        total_loss_val += total_loss.item()

    avg_loss = total_loss_val / len(train_loader)
    epoch_time = time.time() - start_time

    print(f"Epoch {epoch + 1}/{EPOCHS} - Loss: {avg_loss:.4f} - Time: {epoch_time:.2f}")

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
    torch.save(best_model_state, "models/wvs_conv1d_classifier_mr.pth")
    print("Model saved to conv1d_classifier_mr.pth")
else:
    torch.save(model.state_dict(), "../models/wvs_conv1d_classifier_last.pth")
    print("Model saved to conv1d_classifier_last.pth (no improvement)")