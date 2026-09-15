from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import torch
from .utils import visualize


class GCN(torch.nn.Module):
    def __init__(self, dataset, hidden_channels):
        super().__init__()
        torch.manual_seed(1234567)
        self.conv1 = GCNConv(dataset.num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
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


def test(model, data):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    test_correct = pred[data.test_mask] == data.y[data.test_mask]
    test_acc = int(test_correct.sum()) / int(data.test_mask.sum())
    return test_acc


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

    model = GCN(dataset, hidden_channels=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()
    print(model)

    model.eval()
    out = model(data.x, data.edge_index)
    visualize(out, color=data.y)

    for epoch in range(100):
        loss = train(model, data, optimizer, criterion)
        print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}")

    test_acc = test(model, data)
    print(f"Test Accuracy: {test_acc:.4f}")

    model.eval()
    out = model(data.x, data.edge_index)
    visualize(out, color=data.y)


if __name__ == "__main__":
    run()
