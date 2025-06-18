import torch
import torch.nn as nn
from torch_geometric.nn import HeteroConv, GATConv

class OntologyGNNEncoder(nn.Module):
    def __init__(self, in_channels=1, hidden_channels=32, out_channels=32):
        super(OntologyGNNEncoder, self).__init__()
        # self-loop 금지!
        self.hetero_conv1 = HeteroConv({
            ('variable', 'has_value', 'patient'): GATConv(in_channels, hidden_channels, add_self_loops=False),
            ('concept', 'belongs_to', 'variable'): GATConv(in_channels, hidden_channels, add_self_loops=False),
            # 필요시 역방향 edge도 동일하게 추가/수정
            ('variable', 'has_value_rev', 'patient'): GATConv(in_channels, hidden_channels, add_self_loops=False),
            ('variable', 'belongs_to_rev', 'concept'): GATConv(in_channels, hidden_channels, add_self_loops=False),
        }, aggr='mean')
        self.hetero_conv2 = HeteroConv({
            ('variable', 'has_value', 'patient'): GATConv(hidden_channels, out_channels, add_self_loops=False),
            ('concept', 'belongs_to', 'variable'): GATConv(hidden_channels, out_channels, add_self_loops=False),
            ('variable', 'has_value_rev', 'patient'): GATConv(hidden_channels, out_channels, add_self_loops=False),
            ('variable', 'belongs_to_rev', 'concept'): GATConv(hidden_channels, out_channels, add_self_loops=False),
        }, aggr='mean')

    def forward(self, data):
        x_dict, edge_index_dict = data.x_dict, data.edge_index_dict
        x_dict = self.hetero_conv1(x_dict, edge_index_dict)
        x_dict = {k: torch.relu(v) for k, v in x_dict.items()}
        x_dict = self.hetero_conv2(x_dict, edge_index_dict)
        x_dict = {k: torch.relu(v) for k, v in x_dict.items()}
        # patient node embedding만 반환
        return x_dict['patient']
