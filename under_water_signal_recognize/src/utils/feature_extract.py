import random

import scipy.io


import os
import numpy as np
import librosa
from typing import Tuple

from under_water_signal_recognize.src.utils.get_w_mel_feature import W_melspec


def load_mat_file_into_numpy(file_path_data, file_path_label):
    mat = scipy.io.loadmat(file_path_data)
    data = mat['F'].transpose(0, 2, 1)

    mat = scipy.io.loadmat(file_path_label)
    label = mat['ans']
    return data, label


def build_balanced_dataset(
    root_dir: str = '..//data//DeepShip',
    L_w: int = 960,
    step: int = 100,
    target_fs: int = 32000,
    num_per_class: int = 500,
    mode = "train"
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
        if mode == "train":
            folder_path = os.path.join(root_dir, "train", class_name)
        else:
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
                sorted_W = W_melspec(y, L_w=L_w, step=step, fs=sr)[0]

                # if sorted_W.shape != (199, 310):
                #     print(f"Skipped {fname} in {class_name}: shape {sorted_W.shape}")
                #     continue

                all_features.append(sorted_W)
                all_labels.append(label)
                count += 1
            except Exception as e:
                print(f"Failed to process {fname} in {class_name}: {type(e).__name__} - {e}")

        print(f"Collected {count} valid samples for class '{class_name}'")

    features = np.stack(all_features)
    labels = np.array(all_labels, dtype=np.int64)
    return features.astype(np.float32), labels


def normalized_covariance_matrix(D):
    """
    计算归一化的协方差矩阵（相关系数矩阵），即每个元素除以对应特征列的二范数乘积。

    参数:
        D (np.ndarray): 输入数据矩阵，形状为 (N, J)，N 为样本数，J 为特征数。

    返回:
        np.ndarray: 归一化后的协方差矩阵，形状为 (J, J)。
    """
    # 计算未归一化的协方差矩阵
    C = D.T @ D

    # 计算每列的二范数（L2 norm）
    norms = np.linalg.norm(D, axis=0)

    # 构造范数乘积矩阵
    norm_matrix = np.outer(norms, norms)

    # 防止除以0
    norm_matrix[norm_matrix == 0] = 1e-10

    # 归一化协方差矩阵
    C_normalized = C / norm_matrix

    return C_normalized


def compute_MR(A: np.ndarray, alpha: float = 2.0) -> float:
    """
    计算归一化的 M_R 值。

    参数:
        A (np.ndarray): 输入归一化协方差矩阵 (J, J)
        alpha (float): 指数参数，要求 alpha > 0 且 alpha ≠ 1

    返回:
        float: 归一化度量 M_R
    """
    if alpha <= 0 or alpha == 1:
        raise ValueError("alpha must be > 0 and ≠ 1")

    J = A.shape[0]

    # 矩阵 A 的 α 次幂
    A_alpha = np.linalg.matrix_power(A, int(alpha))

    # 计算 Tr(A^α) 和 Tr(A)
    tr_A_alpha = np.trace(A_alpha)
    tr_A = np.trace(A)

    # 计算 M_R
    numerator = np.log(tr_A_alpha) - alpha * np.log(tr_A)
    denominator = np.log(J ** (alpha - 1))
    MR = 1 + numerator / denominator

    return MR



if __name__ == '__main__':
    model = "test"
    features, labels = build_balanced_dataset(mode=model)
    np.savez_compressed(f"..//data//DeepShip//npz//deepship_{model}ed_dataset_window.npz", features=features, labels=labels)





