import torch
import torch.nn as nn

class SpiroPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, age, sex, smoking, concavity):
        # 모두 [B,1] shape 가정
        x = torch.cat([age, sex, smoking, concavity], dim=1)  # [B, 3+4]
        return self.mlp(x).squeeze(1)
