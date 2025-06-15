import librosa
import numpy as np
from scipy import signal


def W_melspec(y, L_w, step, fs):
    """
    Equivalent of the MATLAB function W_melspec.
    
    Parameters:
    y    -- Input signal (1D numpy array)
    L_w  -- Window length (int)
    step -- Step size (int)
    fs   -- Sampling frequency (not used in current logic)
    
    Returns:
    sorted_W -- Sorted and normalized frequency components
    W        -- Raw FFT magnitude components
    """
    num_windows = (len(y) - 1 - L_w) // step
    W = []
    sorted_specs = []

    for i in range(num_windows):
        start = i * step
        window = y[start: start + L_w]
        window = (window - np.mean(window))
        window /= np.linalg.norm(window)
        W1 = np.abs(np.fft.fft(window, 2048))
        W.append(W1[1:200])  # MATLAB 2:200 == Python 1:200 (0-indexed)
    window_spec = np.array(W).T  # Transpose to match W(:,i) in MATLAB
    sorted_window_spec = sorted_spec(window_spec)
    sorted_specs.append(sorted_window_spec)

    stft_spec = signal.stft(y, fs=fs, window='hann', nperseg=L_w, noverlap=L_w - step, nfft=1024)
    stft_spec =stft_spec[-1][1:200][:].T
    sorted_stft_spec = sorted_spec(librosa.power_to_db(np.abs(stft_spec), ref=np.max))
    sorted_specs.append(sorted_stft_spec)

    return sorted_specs


def sorted_spec(W):
    n = W.shape[1]
    Uniform_variable = np.arange(1, n + 1) / n
    sorted_W = np.zeros_like(W)
    for j in range(W.shape[0]):
        sorted_row = np.sort(W[j, :])
        sorted_W[j, :] = sorted_row - Uniform_variable
    return sorted_W
