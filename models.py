import torch
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.nn import TransformerConv

class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)

        x = global_mean_pool(x, batch)
        x = self.fc(x)

        return x


class GraphTransformer(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads=4):
        super(GraphTransformer, self).__init__()
        self.conv1 = TransformerConv(input_dim, hidden_dim, heads=num_heads, dropout=0.1)
        self.conv2 = TransformerConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, dropout=0.1)
        self.fc = torch.nn.Linear(hidden_dim * num_heads, output_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)

        x = global_mean_pool(x, batch)
        x = self.fc(x)

        return x