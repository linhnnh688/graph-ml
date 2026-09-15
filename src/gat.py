from torch_geometric.nn import GATConv
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
import torch.nn.functional as F
import torch
from .utils import visualize
import matplotlib.pyplot as plt
import numpy as np


class GAT(torch.nn.Module):
    def __init__(self, dataset, hidden_channels, heads):
        super().__init__()
        torch.manual_seed(1234567)
        self.conv1 = GATConv(dataset.num_features, hidden_channels, heads)
        self.conv2 = GATConv(heads * hidden_channels, dataset.num_classes, heads)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return x


def train(model, data, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss


def test(model, data, mask):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    correct = pred[mask] == data.y[mask]
    acc = int(correct.sum()) / int(mask.sum())
    return acc


def evaluate(model, data, val_acc_all, test_acc_all):
    plt.figure(figsize=(12, 8))
    plt.plot(
        np.arange(1, len(val_acc_all) + 1),
        val_acc_all,
        label="Validation accuracy",
        c="blue",
    )
    plt.plot(
        np.arange(1, len(test_acc_all) + 1),
        test_acc_all,
        label="Testing accuracy",
        c="red",
    )
    plt.xlabel("Epochs")
    plt.ylabel("Accurarcy")
    plt.title("GATConv")
    plt.legend(loc="lower right", fontsize="x-large")
    plt.savefig("gat_loss.png")
    plt.show()

    model.eval()

    out = model(data.x, data.edge_index)
    visualize(out, color=data.y)


def run():
    dataset = Planetoid(
        root="data/Planetoid", name="Cora", transform=NormalizeFeatures()
    )

    print(f"Dataset: {dataset}:")
    print("======================")
    print(f"Number of graphs: {len(dataset)}")
    print(f"Number of features: {dataset.num_features}")
    print(f"Number of classes: {dataset.num_classes}")

    data = dataset[0]  # Get the first graph object.
    print(data)

    model = GAT(dataset, hidden_channels=8, heads=8)
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()

    val_acc_all = []
    test_acc_all = []

    for epoch in range(1, 101):
        loss = train(model, data, optimizer, criterion)
        val_acc = test(model, data, data.val_mask)
        test_acc = test(model, data, data.test_mask)
        val_acc_all.append(val_acc)
        test_acc_all.append(test_acc)
        print(
            f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}"
        )

    evaluate(model, data, val_acc_all, test_acc_all)
