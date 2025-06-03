import numpy as np


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

    for i in range(num_windows):
        start = i * step
        window = y[start: start + L_w]
        window = (window - np.mean(window))
        window /= np.linalg.norm(window)
        W1 = np.abs(np.fft.fft(window, 2048))
        W.append(W1[1:200])  # MATLAB 2:200 == Python 1:200 (0-indexed)

    W = np.array(W).T  # Transpose to match W(:,i) in MATLAB

    n = W.shape[1]
    Uniform_variable = np.arange(1, n + 1) / n

    sorted_W = np.zeros_like(W)
    for j in range(W.shape[0]):
        sorted_row = np.sort(W[j, :])
        sorted_W[j, :] = sorted_row - Uniform_variable

    return sorted_W
