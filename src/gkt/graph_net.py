import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GCNConv, GATConv, SAGEConv


class GCN(torch.nn.Module):
    def __init__(self, num_skills, hidden_dim=128):
        """
        Represents a simple 3-layer Graph Convolutional Network (GCN)
        with embedding and hidden dimension of hidden_dim.
        """
        super().__init__()
        self.tag = "GCN"
        self.pre_embs = nn.Embedding(num_skills, hidden_dim)
        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.prelu1 = nn.PReLU()
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.prelu2 = nn.PReLU()
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.prelu3 = nn.PReLU()
        self.out = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, edge_index, edge_weight):
        """
        Runs a forward pass through the GCN with given initial skill IDs and
        edge_index and edge_weights.

        Arguments:
          - x: skill IDs (torch.Tensor)
          - edge_index: edges in skill graph (torch.Tensor)
          - edge_weight: edge weights of skill graph (torch.Tensor)

        Returns:
          - final node embedding for skill
        """
        h0 = self.pre_embs(x)
        h1 = self.dropout(
            self.prelu1(self.conv1(h0, edge_index, edge_weight=edge_weight))
        )
        h2 = self.dropout(
            self.prelu2(self.conv2(h1, edge_index, edge_weight=edge_weight))
        )
        h3 = self.prelu3(self.conv3(h2, edge_index, edge_weight=edge_weight))
        return self.out(h3)


class GraphSAGE(torch.nn.Module):
    def __init__(self, num_skills, hidden_dim=128):
        """
        Represents a 3-layer GraphSAGE GNN model
        with embedding and hidden dimension of hidden_dim.
        """
        super().__init__()
        self.tag = "GraphSAGE"
        self.pre_embs = nn.Embedding(num_skills, hidden_dim)
        self.conv1 = SAGEConv(hidden_dim, hidden_dim)
        self.prelu1 = nn.PReLU()
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.prelu2 = nn.PReLU()
        self.conv3 = SAGEConv(hidden_dim, hidden_dim)
        self.prelu3 = nn.PReLU()
        self.out = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, edge_index, edge_weight):
        """
        Runs a forward pass through GraphSAGE with given initial skill IDs and
        edge_index and edge_weights.

        Arguments:
          - x: skill IDs (torch.Tensor)
          - edge_index: edges in skill graph (torch.Tensor)
          - edge_weight: edge weights of skill graph (torch.Tensor)

        Returns:
          - final node embedding for skill
        """
        h0 = self.pre_embs(x)
        h1 = self.dropout(self.prelu1(self.conv1(h0, edge_index)))
        h2 = self.dropout(self.prelu2(self.conv2(h1, edge_index)))
        h3 = self.prelu3(self.conv3(h2, edge_index))
        return self.out(h3)


class GAT(torch.nn.Module):
    def __init__(self, num_skills, hidden_dim=128):
        """
        Represents a 3-layer Graph Attention Network (GAT)
        with embedding and hidden dimension of hidden_dim.
        """
        super().__init__()
        self.tag = "GAT"
        self.pre_embs = nn.Embedding(num_skills, hidden_dim)
        self.conv1 = GATConv(hidden_dim, hidden_dim)
        self.prelu1 = nn.PReLU()
        self.conv2 = GATConv(hidden_dim, hidden_dim)
        self.prelu2 = nn.PReLU()
        self.conv3 = GATConv(hidden_dim, hidden_dim)
        self.prelu3 = nn.PReLU()
        self.out = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, edge_index, edge_weight):
        """
        Runs a forward pass through GAT with given initial skill IDs and
        edge_index and edge_weights.

        Arguments:
          - x: skill IDs (torch.Tensor)
          - edge_index: edges in skill graph (torch.Tensor)
          - edge_weight: edge weights of skill graph (torch.Tensor)

        Returns:
          - final node embedding for skill
        """
        h1 = self.dropout(self.prelu1(self.conv1(self.pre_embs(x), edge_index)))
        h2 = self.dropout(self.prelu2(self.conv2(h1, edge_index)))
        h3 = self.prelu3(self.conv3(h2, edge_index))
        return self.out(h3)
