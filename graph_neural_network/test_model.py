# %% Import the model
import logging
logging.basicConfig(level=logging.INFO)
logging.info('Importing Dataset')

import os
import torch
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

current_directory = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_directory, 'model.pth')

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

model = torch.load(file_path)
model.eval()


# %% Generate Environment

logging.info('Generating Environment')

import numpy as np
from planning_sandbox.environment_class import Environment

size = 32
num_skills = 2 # can not change for now
num_agents= 3
num_goals= 5
use_geo_data= False, # can not change for now
assume_lander= False,
random_map= False # can not change for now

env = Environment(size=size, num_skills=num_skills, num_agents=num_agents, num_goals=num_goals, use_geo_data=use_geo_data, assume_lander=assume_lander, random_map=random_map)

# %% Generate Graph
from torch_geometric.data import Data
from utils import compare_observation_and_solution_graph


logging.info('Generating Graphs')
observation_graph = env.get_observation_graph() # nx graph
env.find_numerical_solution(solve_type='optimal')
solution_graph = env.get_solution_graph() # nx graph

features = []
edge_index = [[],[]]
edge_attr = []
y = []
edge_index_y = [[],[]]
edge_attr_y = []

for node in observation_graph.nodes(data=True):
    node_features = []
    node_features.append(node[1]['type'])
    node_features.append(node[1]['x'])
    node_features.append(node[1]['y'])
    node_features.extend(node[1]['skills'])
    features.append(node_features)

for edge in observation_graph.edges(data=True):
    attrs = []
    edge_index[0].append(edge[0])
    edge_index[1].append(edge[1])
    attrs.append(edge[2]['idx'])
    attrs.append(edge[2]['manhattan_distance'])
    edge_attr.append(attrs)

for edge in observation_graph.edges():
    if edge in solution_graph.edges():
        y.append(solution_graph.edges[edge]['idx'])
    else:
        y.append(0)

for edge in solution_graph.edges(data=True):
    attrs = []
    edge_index_y[0].append(edge[0])
    edge_index_y[1].append(edge[1])
    attrs.append(edge[2]['idx'])
    attrs.append(edge[2]['manhattan_distance'])
    edge_attr_y.append(attrs)

edge_index = torch.tensor(edge_index, dtype=torch.long)
edge_attr = torch.tensor(edge_attr, dtype=torch.float)
edge_attr_y = torch.tensor(edge_attr_y, dtype=torch.float)
edge_index_y = torch.tensor(edge_index_y, dtype=torch.long)
y = torch.tensor(y, dtype=torch.float)

x = torch.tensor(features, dtype=torch.float)

data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, edge_index_y=edge_index_y, edge_attr_y=edge_attr_y)

compare_observation_and_solution_graph(observation_graph, solution_graph)

# %% Decode and Encode

z = model.encode(data.x, data.edge_index)
pred = model.decode(z, data.edge_index).view(-1)
pred_prob = torch.sigmoid(pred)
threshold = 0.5
pred_label = (pred_prob > threshold).float()


# %%
