import torch

def compute_concavity_features(flow_patches, patch_len):
    """
    flow_patches: [B, max_patch, 1, patch_len]
    return: [B, 4]
    """
    B = flow_patches.size(0)
    features = []
    for i in range(B):
        patches = flow_patches[i]  # [max_patch, 1, patch_len]
        diff = patches[:, :, 1:] - patches[:, :, :-1]
        diff2 = diff[:, :, 1:] - diff[:, :, :-1]
        max_c = diff2.max(dim=2).values.mean(dim=0)
        min_c = diff2.min(dim=2).values.mean(dim=0)
        mean_c = diff2.mean(dim=2).mean(dim=0)
        std_c = diff2.std(dim=2).mean(dim=0)
        feat = torch.cat([max_c, min_c, mean_c, std_c], dim=0)  # [4]
        features.append(feat)
    return torch.stack(features)  # [B,4]
