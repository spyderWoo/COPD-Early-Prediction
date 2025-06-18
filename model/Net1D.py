import torch
import torch.nn as nn

class FusionPredictor(nn.Module):
    def __init__(self, spiro_dim, graph_dim, hidden_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(spiro_dim + graph_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        
    def forward(self, spiro_output, graph_output):
        combined = torch.cat([spiro_output, graph_output], dim=1)
        return self.fc(combined).squeeze(1)