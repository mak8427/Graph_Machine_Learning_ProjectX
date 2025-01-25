import torch
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.nn import TransformerConv
import torch.nn.functional as F
import torch.nn as nn
from torch_scatter import scatter_mean

class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index, num_nodes, batch = data.x, data.edge_index, data.num_nodes, data.batch

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
    

class GINELayer(nn.Module):
    def __init__(self, in_channels, out_channels, activation=F.relu, use_batch_norm=True):
        super(GINELayer, self).__init__()
        self.activation = activation
        # initialize MLPs for nodes and edges
        # self.node_mlp = nn.Linear(in_channels, out_channels)
        self.node_mlp = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )
        self.edge_mlp = nn.Linear(1, in_channels)
        # trainable epsilon and batch norm
        self.eps = nn.Parameter(torch.Tensor([0]))
        self.use_batch_norm = use_batch_norm
        
        if self.use_batch_norm:
            self.norm = nn.BatchNorm1d(out_channels)
        else:
            self.norm = nn.LayerNorm(out_channels)

        # skip connection
        self.residual_proj = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index, edge_attr):
        src, dst = edge_index

        # updating edge features with mlp, bringing them to the same dimension as node features
        edge_attr = self.edge_mlp(edge_attr.float())
        message = x[src] + edge_attr

        # aggregating messages
        aggr_out = scatter_mean(message, dst, dim=0, dim_size=x.size(0))

        # node update
        out = self.node_mlp((1 + self.eps) * x + aggr_out)
        
        # skip connection
        out += self.residual_proj(x.float())

        # apply normalization
        if self.use_batch_norm:
            out = self.norm(out)  

        out = self.activation(out)

        out = F.dropout(out, p=0.2, training=self.training)
        
        return out

class GINE(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_channels=64, num_layers=4, use_batch_norm=True):
        super(GINE, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.use_batch_norm = use_batch_norm

        # stacking up GINE layers
        self.layers = nn.ModuleList()
        self.layers.append(GINELayer(num_features, hidden_channels, use_batch_norm=use_batch_norm))
        
        for _ in range(num_layers - 1):
            self.layers.append(GINELayer(hidden_channels, hidden_channels, use_batch_norm=use_batch_norm))

        self.linear = nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, edge_attr, batch):
        # Pass through GINE layers
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)

        # global pooling (mean pooling) to obtain graph-level embeddings
        x = global_mean_pool(x, batch) 

        # final linear layer for graph-level output
        x = self.linear(x)
        return x
