import os
import numpy as np
import torch
from madskills.environment.environment_class import Environment
from torch_geometric.data import Data
from madskills.machine_learning.gnn_solver.utils import MyDataset, compare_observation_and_solution_graph


class GraphGenerator():
    def __init__(self,
                 save_path,
                 map_size_range=[32,100],
                 num_skill_range=[2,2],
                 num_agents_range=[1,3],
                 num_goals_range=[2,5],
                 use_geo_data=True,
                 assume_lander=False,
                 random_maps=True,
                 save_interval=50,
                 overwrite_protection=True
                 ):
        self.save_path = save_path
        self.save_interval = save_interval
        self.map_size_range = map_size_range
        self.num_skill_range = num_skill_range
        self.num_agents_range = num_agents_range
        self.num_goals_range = num_goals_range
        self.use_geo_data = use_geo_data
        self.assume_lander = assume_lander
        self.random_maps = random_maps
        self.env = None
        self.data_list = []
        self.overwrite_protection = overwrite_protection
        self.check_existing_data()

    def generate_graphs(self, num_graphs):
        for i in range(num_graphs):
            print(f"Generating graph {len(self.data_list)+1}...", end='\r')
            
            size = np.random.randint(self.map_size_range[0], self.map_size_range[1]+1)
            num_skills = np.random.randint(self.num_skill_range[0], self.num_skill_range[1]+1)
            num_agents= np.random.randint(self.num_agents_range[0], self.num_agents_range[1]+1)
            num_goals= np.random.randint(self.num_goals_range[0], self.num_goals_range[1]+1)
            use_geo_data = self.use_geo_data
            assume_lander = self.assume_lander
            random_map = self.random_maps

            if not self.env:
                self.env = Environment(size=size, num_skills=num_skills, num_agents=num_agents, num_goals=num_goals, use_geo_data=use_geo_data, assume_lander=assume_lander, random_map=random_map)
            else:
                self.env.reset(num_agents=num_agents, num_goals=num_goals)

            observation_graph = self.env.get_observation_graph() # nx graph

            self.env.find_numerical_solution(solve_type='optimal')

            solution_graph = self.env.get_solution_graph() # nx graph

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

            self.data_list.append(data)

            if len(self.data_list) % self.save_interval == 0 or i == self.save_interval-1:
                self.save_data()

    def check_existing_data(self):
        if os.path.exists(self.save_path):
            if self.overwrite_protection:
                overwrite = input(f"File '{self.save_path}' already exists. Overwrite? (y/n): ")
                if overwrite.lower() != 'y':
                    new_filename = input("Enter new filename: ")
                    self.save_path = new_filename
                    self.overwrite_protection = False
                    return

    def save_data(self):
        print(f"Saving dataset to '{self.save_path}'...")
        print(f"Overwrite protection: {self.overwrite_protection}")

        self.check_existing_data()
            
        self.overwrite_protection = False
        dataset = MyDataset(root=self.save_path, data_list=self.data_list)
        dataset.process()

