import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from under_water_signal_recognize.src.conv1d_classifier import Conv1DRowWiseClassifier
from under_water_signal_recognize.src.utils import load_mat_file_into_numpy

data_path = '../data/Wmel_Feature.mat'
label_path = '../data/Label.mat'
features, labels = load_mat_file_into_numpy(data_path, label_path)

features = features.astype(np.float32)
labels = labels.squeeze().astype(np.int64) - 1

X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=30, random_state=42, stratify=labels
)

class SignalDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X).unsqueeze(1)
        self.y = torch.tensor(y)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(SignalDataset(X_train, y_train), batch_size=16, shuffle=True)
test_loader = DataLoader(SignalDataset(X_test, y_test), batch_size=10)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Conv1DRowWiseClassifier(num_rows=X_train.shape[1], num_classes=4).to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(10):
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
    print(f"Epoch {epoch+1} | Loss: {total_loss:.4f}")

model.eval()
correct = 0
total = 0
all_preds = []
all_labels = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        output = model(X_batch)
        preds = torch.argmax(output, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())

        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)

print(f"\nTest Accuracy: {correct / total * 100:.2f}%\n")

print("Test Set Predictions vs Labels:")
for i, (pred, label) in enumerate(zip(all_preds, all_labels)):
    print(f"Sample {i:2d} | Predicted: {pred} | Label: {label}")
