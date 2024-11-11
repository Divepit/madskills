import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv
from torch_geometric.explain import Explainer, GNNExplainer
from utils import MyDataset  # Assuming 'utils.py' contains your dataset class

# Set up the dataset path and load it
current_directory = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(current_directory, 'graphs', 'dataset_2sk.npy')
dataset = MyDataset(root=dataset_path)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

print(f"Dataset size: {len(dataset)}")
example = dataset[0]
example.validate(raise_on_error=True)
print("Sample Graph Data:", example)
print("Node Features:", example['x'])
print("Edge Index:", example['edge_index'])


class LinkPredictionAndRegressionModel(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        # Shared GNN layers
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)

        # Output head for link prediction (classification)
        self.link_pred_head = nn.Sequential(
            nn.Linear(2 * hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1),
            nn.Sigmoid()  # Output: probability (0 to 1)
        )
        
        # Output head for link regression (predicting 'idx')
        self.link_reg_head = nn.Sequential(
            nn.Linear(2 * hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1)  # Output: continuous value for 'idx'
        )

    def forward(self, x, edge_index):
        # Forward pass through GNN layers
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)

        # Generate edge embeddings
        def get_edge_embeddings(edge_index, node_embeddings):
            src, dst = edge_index
            edge_features = torch.cat([node_embeddings[src], node_embeddings[dst]], dim=-1)
            return edge_features

        edge_embeddings = get_edge_embeddings(edge_index, x)

        # Link prediction (classification)
        link_pred = self.link_pred_head(edge_embeddings)

        # Link regression (predicting 'idx')
        link_reg = self.link_reg_head(edge_embeddings)

        return link_pred, link_reg
    

# Hyperparameters
in_channels = dataset[0].x.shape[1]
hidden_channels = 32
num_epochs = 10
alpha = 0.5  # Weight for link prediction loss
beta = 0.5   # Weight for link regression loss

# Initialize model, optimizer, and loss functions
model = LinkPredictionAndRegressionModel(in_channels, hidden_channels)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Create a DataLoader
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for data in loader:
        optimizer.zero_grad()
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # Forward pass
        link_pred, link_reg = model(x, edge_index)

        # Extract the ground truth for link prediction and regression
        true_labels = torch.ones(edge_index.shape[1], 1, device=x.device)  # Assuming all edges in edge_index are positive
        true_weights = edge_attr[:, 0].view(-1, 1)  # Assuming the first attribute is 'idx'

        # Compute link prediction loss (binary cross-entropy)
        link_pred_loss = F.binary_cross_entropy(link_pred, true_labels)

        # Compute link regression loss (mean squared error)
        link_reg_loss = F.mse_loss(link_reg, true_weights)

        # Combine the losses
        loss = alpha * link_pred_loss + beta * link_reg_loss
        total_loss += loss.item()

        # Backpropagation
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss:.4f}")

print("Training complete!")

model.eval()
with torch.no_grad():
    for i, data in enumerate(dataset[:5]):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        link_pred, link_reg = model(x, edge_index)

        print(f"Sample {i + 1}")
        print("Predicted Link Existence:", link_pred.round())
        print("Predicted Weights (idx):", link_reg)
        print("True Weights (idx):", edge_attr[:, 0])
        print()