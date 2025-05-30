import numpy as np
import torch
from torch.utils.data import DataLoader
from under_water_signal_recognize.src.conv1d_classifier import Conv1DRowWiseClassifier
from under_water_signal_recognize.src.dataset import SignalDataset

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------- 功能函数 ----------------------

def print_classification_results(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\nOverall Accuracy: {accuracy * 100:.2f}%")

    target_names = ['Cargo (0)', 'Passengership (1)', 'Tanker (2)', 'Tug (3)']
    print("\nClassification Report (per class):")
    print(classification_report(y_true, y_pred, target_names=target_names, digits=4))

    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))


def plot_classification_results(y_true, y_pred):
    target_names = ['Cargo', 'Passengership', 'Tanker', 'Tug']
    cm = confusion_matrix(y_true, y_pred)

    # 混淆矩阵热图
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()

    # 精确率、召回率、F1条形图
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1, 2, 3])
    x = np.arange(len(target_names))
    width = 0.25

    plt.figure(figsize=(8, 5))
    plt.bar(x - width, precision, width, label='Precision')
    plt.bar(x, recall, width, label='Recall')
    plt.bar(x + width, f1, width, label='F1-Score')

    plt.xticks(x, target_names)
    plt.ylim(0, 1)
    plt.ylabel('Score')
    plt.title('Per-Class Evaluation Metrics')
    plt.legend()
    plt.tight_layout()
    plt.show()

# ---------------------- 主流程 ----------------------

# 加载 .npz 文件数据
data = np.load('../data/DeepShip/Test/deepship_dataset.npz')
features = data['features']  # shape: (2000, 199, 310)
labels = data['labels']      # shape: (2000,)

# 创建测试集加载器
test_loader = DataLoader(SignalDataset(features, labels), batch_size=32)

# 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Conv1DRowWiseClassifier(num_rows=features.shape[1], num_classes=4).to(device)
model.load_state_dict(torch.load("conv1d_classifier.pth", map_location=device))
model.eval()

# 推理
all_preds = []
all_labels = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        output = model(X_batch)
        preds = torch.argmax(output, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())

# 输出结果与可视化
print_classification_results(all_labels, all_preds)
plot_classification_results(all_labels, all_preds)
