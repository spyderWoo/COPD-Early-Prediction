
# preprocessing/smoother.py

import numpy as np
from scipy.ndimage import gaussian_filter1d

def gaussian_smooth(signal, sigma=2.0):
    """
    Gaussian smoothing 적용
    """
    return gaussian_filter1d(signal, sigma=sigma)

# 선택적으로 기존 함수도 유지
def smooth_signal(signal, method='gaussian', sigma=2.0):
    if method == 'gaussian':
        return gaussian_smooth(signal, sigma)
    else:
        return signal
