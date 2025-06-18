import torch
import torch.nn as nn

def compute_concavity_features(flow_patches, patch_length):
    # dummy stub: 실제로는 4상위 concavity 계산
    B,_,_,L = flow_patches.shape
    return torch.zeros((B,4), dtype=torch.float32)

class SpiroPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, age, sex, smoking, concav):
        x = torch.cat([age, sex, smoking, concav], dim=1)
        x = self.act(self.fc1(x))
        return self.fc2(x).squeeze(1)
