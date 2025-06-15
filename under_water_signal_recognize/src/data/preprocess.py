import scipy.io
import os
import numpy as np
import librosa
from typing import Tuple
from under_water_signal_recognize.src.features.get_w_mel_feature import W_melspec


def load_mat_file_into_numpy(file_path_data, file_path_label):
    mat = scipy.io.loadmat(file_path_data)
    data = mat['F'].transpose(0, 2, 1)
    label = scipy.io.loadmat(file_path_label)['ans']
    return data, label


def build_test_dataset_from_wav_folder(folder_path: str, L_w=960, step=100, target_fs=32000) -> Tuple[np.ndarray, np.ndarray]:
    features_list = []
    for fname in os.listdir(folder_path):
        if fname.endswith('.wav'):
            file_path = os.path.join(folder_path, fname)
            try:
                y, sr = librosa.load(file_path, sr=target_fs)
                sorted_W = W_melspec(y, L_w=L_w, step=step)
                features_list.append(sorted_W)
            except Exception as e:
                print(f"Skipping {fname}: {e}")
    features = np.stack(features_list)
    labels = np.zeros(len(features), dtype=np.int64)
    return features.astype(np.float32), labels


def build_balanced_dataset(root_dir: str, L_w=960, step=100, target_fs=32000, num_per_class=500) -> Tuple[np.ndarray, np.ndarray]:
    class_map = {'Cargo': 0, 'Passengership': 1, 'Tanker': 2, 'Tug': 3}
    all_features, all_labels = [], []

    for class_name, label in class_map.items():
        folder_path = os.path.join(root_dir, class_name, "Test")
        wav_files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]
        np.random.shuffle(wav_files)
        count = 0
        for fname in wav_files:
            if count >= num_per_class:
                break
            try:
                y, sr = librosa.load(os.path.join(folder_path, fname), sr=target_fs)
                W = W_melspec(y, L_w, step)
                if W.shape != (199, 310):
                    print(f"Skipped {fname} in {class_name}: shape {W.shape}")
                    continue
                all_features.append(W)
                all_labels.append(label)
                count += 1
            except Exception as e:
                print(f"Failed to process {fname} in {class_name}: {type(e).__name__} - {e}")
        print(f"Collected {count} valid samples for '{class_name}'")

    features = np.stack(all_features)
    labels = np.array(all_labels, dtype=np.int64)
    return features.astype(np.float32), labels
