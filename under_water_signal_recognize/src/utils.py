import random

import scipy.io


import os
import numpy as np
import librosa
from typing import Tuple

from under_water_signal_recognize.src.get_w_mel_feature import W_melspec


def load_mat_file_into_numpy(file_path_data, file_path_label):
    mat = scipy.io.loadmat(file_path_data)
    data = mat['F'].transpose(0, 2, 1)

    mat = scipy.io.loadmat(file_path_label)
    label = mat['ans']
    return data, label


def build_test_dataset_from_wav_folder(folder_path: str, L_w: int=960, step: int=100, target_fs: int=32000) -> Tuple[np.ndarray, np.ndarray]:
    features_list = []

    for fname in os.listdir(folder_path):
        if fname.endswith('.wav'):
            file_path = os.path.join(folder_path, fname)
            try:
                # 读取音频，自动重采样
                y, sr = librosa.load(file_path, sr=target_fs)
                sorted_W = W_melspec(y, L_w=L_w, step=step, fs=sr)
                features_list.append(sorted_W)
            except Exception as e:
                print(f"❌ Skipping {fname}: {e}")

    features = np.stack(features_list)  # (n, 310, 199)
    labels = np.zeros(len(features), dtype=np.int64)
    return features.astype(np.float32), labels


def build_balanced_dataset(
    root_dir: str = '..//data//DeepShip',
    L_w: int = 960,
    step: int = 100,
    target_fs: int = 32000,
    num_per_class: int = 500
) -> Tuple[np.ndarray, np.ndarray]:
    class_map = {
        'Cargo': 0,
        'Passengership': 1,
        'Tanker': 2,
        'Tug': 3
    }

    all_features = []
    all_labels = []

    for class_name, label in class_map.items():
        folder_path = os.path.join(root_dir, class_name, "Test")
        wav_files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]
        random.shuffle(wav_files)

        count = 0
        for fname in wav_files:
            if count >= num_per_class:
                break
            file_path = os.path.join(folder_path, fname)
            try:
                y, sr = librosa.load(file_path, sr=target_fs)
                sorted_W = W_melspec(y, L_w=L_w, step=step, fs=sr)

                if sorted_W.shape != (199, 310):
                    print(f"Skipped {fname} in {class_name}: shape {sorted_W.shape}")
                    continue

                all_features.append(sorted_W)
                all_labels.append(label)
                count += 1
            except Exception as e:
                print(f"Failed to process {fname} in {class_name}: {type(e).__name__} - {e}")

        print(f"Collected {count} valid samples for class '{class_name}'")

    features = np.stack(all_features)  # shape: (N, 199, 310)
    labels = np.array(all_labels, dtype=np.int64)
    return features.astype(np.float32), labels


if __name__ == '__main__':
    features, labels = build_balanced_dataset()
    np.savez_compressed("..//data//DeepShip//Test//deepship_dataset.npz", features=features, labels=labels)
    print("✅ Saved dataset to deepship_dataset.npz")



