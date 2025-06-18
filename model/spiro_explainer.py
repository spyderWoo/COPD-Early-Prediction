import torch


def compute_concavity_features(flow_patches, patch_len):
    """유량 패치에서 concavity(2차 차분) 통계치를 계산한다.

    Parameters
    ----------
    flow_patches : torch.Tensor
        ``[B, P, 1, patch_len]`` 형태의 유량 패치 텐서
    patch_len : int
        패치 길이. 현재 계산에는 사용하지 않지만 인터페이스 유지를 위해 남겨둔다.

    Returns
    -------
    torch.Tensor
        각 배치에 대해 최대값, 최소값, 평균, 표준편차를 담은 ``[B, 4]`` 텐서
    """

    # 1차 차분 -> 2차 차분 순으로 계산한다.
    diff1 = flow_patches[..., 1:] - flow_patches[..., :-1]  # (B, P, 1, L-1)
    diff2 = diff1[..., 1:] - diff1[..., :-1]                # (B, P, 1, L-2)

    # 패치 차원(P)에 대해 평균을 취해 단일 시퀀스로 만든다.
    diff2 = diff2.squeeze(2).mean(dim=1)                    # (B, L-2)

    max_val = diff2.max(dim=1).values
    min_val = diff2.min(dim=1).values
    mean_val = diff2.mean(dim=1)
    std_val = diff2.std(dim=1)

    # [B, 4] 형태로 반환
    return torch.stack([max_val, min_val, mean_val, std_val], dim=1)
