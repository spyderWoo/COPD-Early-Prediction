# preprocessing/flow_converter.py

import numpy as np
from .smoother import smooth_signal

def convert_volume_to_flow(volume_signal, dt=0.01):
    """
    부피 시계열 → 유량 시계열로 변환 (단순 finite difference 방식)
    """
    return np.gradient(volume_signal, dt)

def time_to_flow(vol, dt=0.01):
    """
    부피 → 유량 미분 (유량은 단위 시간당 부피 변화량)
    """
    return np.gradient(vol, dt)

def construct_flow_volume(vol, flow):
    """
    vol: [N] → 원래는 smoothed volume
    flow: [M] → flow 값 (길이 다를 수 있음)

    두 벡터를 길이 맞춰서 [N, 2] 형태로 쌍을 이룸
    """
    min_len = min(len(vol), len(flow))
    vol = vol[:min_len]
    flow = flow[:min_len]
    return np.stack([vol, flow], axis=1)