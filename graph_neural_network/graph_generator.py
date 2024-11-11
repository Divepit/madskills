import os
import numpy as np
import torch
from planning_sandbox.environment_class import Environment
from torch_geometric.data import Data
from utils import MyDataset, compare_observation_and_solution_graph

print("[Version check]: v1.1")

save_interval = 10
current_directory = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_directory, 'graphs', 'data_objects_2sk.npy')
dataset_path = os.path.join(current_directory, 'graphs', 'dataset_2sk.npy')

try:
    data_objects = torch.load(file_path, weights_only=False)
    print(f"Loaded {len(data_objects)} data objects from file.")
except FileNotFoundError:
    print(f"No existing file found. Starting fresh.")
    data_objects = []

os.makedirs(os.path.dirname(file_path), exist_ok=True)



while True:
    try:
        print(f"Generating map {len(data_objects)+1}...", end='\r')
        size = 32
        num_skills = 2 # can not change for now
        num_agents= np.random.randint(2, 4)
        num_goals= np.random.randint(3, 6)
        use_geo_data= False, # can not change for now
        assume_lander= False,
        random_map= False # can not change for now


        env = Environment(size=size, num_skills=num_skills, num_agents=num_agents, num_goals=num_goals, use_geo_data=use_geo_data, assume_lander=assume_lander, random_map=random_map)

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
        
        x = torch.tensor(features, dtype=torch.float)
        y = edge_index_y

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, edge_attr_y=edge_attr_y)

        data_objects.append(data)

        if len(data_objects) % save_interval == 0:
            print("\nSaving data to 'data_objects.pt'...")
            torch.save(data_objects, file_path)
            print(f"Saved {len(data_objects)} data objects.")
            print(f"Saving dataset to '{dataset_path}'...")
            os.makedirs(dataset_path, exist_ok=True)
            dataset = MyDataset(root=dataset_path, data_list=data_objects)
            dataset.process()
            print("Dataset saved successfully.")

        # compare_observation_and_solution_graph(observation_graph, solution_graph)

    except KeyboardInterrupt:
        print("\nSaving data to 'data_objects.pt'...")
        torch.save(data_objects, file_path)
        print(f"Saved {len(data_objects)} data objects.")
        print(f"Saving dataset to '{dataset_path}'...")
        os.makedirs(dataset_path, exist_ok=True)
        dataset = MyDataset(root=dataset_path, data_list=data_objects)
        dataset.process()
        print("Dataset saved successfully.")
        break


