# %% Import our Data
import logging
logging.basicConfig(level=logging.INFO)
logging.info('Importing Dataset')

import os
import torch.utils
import torch.utils.data
from utils import MyDataset

current_directory = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_directory, 'graphs', 'data_objects_2sk.npy')
dataset_path = os.path.join(current_directory, 'graphs', 'dataset_2sk.npy')
dataset = MyDataset(root=dataset_path) # Contains areound 37000 PyG Data objects


# %% Set up device
logging.info('Setting up device')

if torch.cuda.is_available():
    device = torch.device('cuda')
# elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
#     device = torch.device('mps')
else:
    device = torch.device('cpu')

logging.info(f"Using device: {device}")

# %% Split the Data into Training and Testing Sets
logging.info('Splitting the data into training and testing sets')

import torch
from torch_geometric.loader import DataLoader

train_split = 0.6
test_split = 0.2
val_split = 0.2

train_size = int(train_split * len(dataset))
test_size = int(test_split * len(dataset))
val_size = len(dataset) - train_size - test_size

train_dataset, test_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, test_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# %% Create the Graph Neural Network with GCN Layers
logging.info('Creating the Graph Neural Network')

from torch_geometric.nn import GCNConv
import torch.nn.functional as F

class Net(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

    def decode(self, z, edge_index):
        scores = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)
        return scores

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()

# %% Create the Loss Function
logging.info('Creating the loss function')

model = Net(dataset[0].x.size(1), 16, 8).to(device)
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
criterion = torch.nn.BCEWithLogitsLoss()

# %% Define training loop with accuracy computation
logging.info('Defining the training loop')

from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score
from torch_geometric.utils import to_undirected

def train():
    model.train()
    total_loss = 0
    total_correct = 0
    total_examples = 0

    for data in train_loader:
        optimizer.zero_grad()
        data = data.to(device)

        # Encode node features using the fully connected graph
        z = model.encode(data.x, data.edge_index)

        # Create labels: 1 if the edge exists in edge_index_y, 0 otherwise
        y = torch.tensor(data.y, dtype=torch.float).dim.float().to(device)

        # Decode the fully connected edges
        pred = model.decode(z, data.edge_index).view(-1)

        # Calculate loss
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Compute accuracy
        pred_prob = torch.sigmoid(pred)
        pred_label = (pred_prob > 0.5).float()
        correct = pred_label.eq(y).sum().item()
        total_correct += correct
        total_examples += y.size(0)

    acc = total_correct / total_examples
    return total_loss / len(train_loader), acc

# %% Define evaluation function
def evaluate(loader):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_examples = 0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            z = model.encode(data.x, data.edge_index)

            y = torch.tensor(data.y, dtype=torch.float).float().to(device)

            # Decode the fully connected edges
            pred = model.decode(z, data.edge_index).view(-1)
            loss = criterion(pred, y)
            total_loss += loss.item()

            # Compute accuracy
            pred_prob = torch.sigmoid(pred)
            pred_label = (pred_prob > 0.5).float()
            correct = pred_label.eq(y).sum().item()
            total_correct += correct
            total_examples += y.size(0)

            y_true.append(y.cpu())
            y_pred.append(pred_prob.cpu())

    acc = total_correct / total_examples
    avg_loss = total_loss / len(loader)
    y_true = torch.cat(y_true)
    y_pred = torch.cat(y_pred)
    auc = roc_auc_score(y_true.numpy(), y_pred.numpy())

    return avg_loss, acc, auc

# %% Train the model with accuracy tracking
logging.info('Training the model')

for epoch in range(1, 30):
    loss, train_acc = train()
    val_loss, val_acc, val_auc = evaluate(val_loader)
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val AUC: {val_auc:.4f}')

# %% Save the model
torch.save(model, current_directory+'/model.pth')