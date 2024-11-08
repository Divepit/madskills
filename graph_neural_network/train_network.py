import os
from utils import MyDataset

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

current_directory = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(current_directory, 'graphs', 'dataset_2sk.npy')

dataset = MyDataset(root=dataset_path)

print(f"Loaded {len(dataset)} data objects from file.")
print(f"Example data object: {dataset[0]}")

class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)
    

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN().to(device)
data = dataset[0].to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()

model.eval()
pred = model(data).argmax(dim=1)
correct = pred.eq(data.y).sum().item()
print(f'Accuracy: {correct / data.num_nodes:.4f}')