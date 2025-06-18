import torch
import torch.nn as nn
import torch.nn.functional as F

class FusionModel(nn.Module):
    def __init__(self, gnn_dim=32, spiro_dim=64, hidden_dim=64):
        super(FusionModel, self).__init__()
        # 입력: [gnn_emb, spiro_emb, age, sex, smoking] concat
        self.input_dim = gnn_dim + spiro_dim + 3  # 3: age, sex, smoking (float)
        self.fc1 = nn.Linear(self.input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 1)

    def forward(self, gnn_emb, spiro_emb, age, sex, smoking):
        # gnn_emb: (B, gnn_dim)
        # spiro_emb: (B, spiro_dim)
        # age, sex, smoking: (B,)
        if age.dim() == 1:
            age = age.unsqueeze(1)
        if sex.dim() == 1:
            sex = sex.unsqueeze(1)
        if smoking.dim() == 1:
            smoking = smoking.unsqueeze(1)
        x = torch.cat([gnn_emb, spiro_emb, age, sex, smoking], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        out = self.fc_out(x)
        return out  # (B, 1)
