import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from under_water_signal_recognize.src.features.get_w_mel_feature import W_melspec

wav_path = "data/DeepShip/train/Cargo/41_1.wav"
signal, sample_rate = librosa.load(wav_path, sr=None)  # sr=None 保持原始采样率

# 显示基本信息
print(f"采样率: {sample_rate} Hz")
print(f"音频时长: {len(signal)/sample_rate:.2f} 秒")
print(f"采样点数: {len(signal)}")

# 绘制波形图
plt.figure(figsize=(14, 5))
librosa.display.waveshow(signal, sr=sample_rate)
plt.title("single")
plt.xlabel("Times (s)")
plt.ylabel("Amplitude")
plt.tight_layout()
plt.show()

# 提取梅尔频谱
mel_spec = librosa.feature.melspectrogram(y=signal, sr=sample_rate)
log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

# 绘制梅尔频谱
plt.figure(figsize=(14, 5))
librosa.display.specshow(log_mel_spec, sr=sample_rate,
                         x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title('mel')
plt.tight_layout()
plt.show()

# 提取W向量谱
W_specs = W_melspec(y=signal, L_w=960, step=100, fs=sample_rate)
# 绘制W向量谱
plt.figure(figsize=(14, 5))
librosa.display.specshow(W_specs[0], sr=sample_rate,
                         x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title('window')
plt.tight_layout()
plt.show()


plt.figure(figsize=(14, 5))
librosa.display.specshow(W_specs[1], sr=sample_rate,
                         x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title('stft')
plt.tight_layout()
plt.show()