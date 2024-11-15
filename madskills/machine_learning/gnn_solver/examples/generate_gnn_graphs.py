import os
from madskills.machine_learning.gnn_solver.graph_generator_class import GraphGenerator

current_directory = os.path.dirname(os.path.abspath(__file__))
dataset_path = current_directory + '/graphs/random_graphs.npy'

gen = GraphGenerator(
    save_path = dataset_path,
    map_size_range = [32, 100],
    num_skill_range = [2, 2],
    num_agents_range = [1, 3],
    num_goals_range = [2, 5],
    use_geo_data = True,
    assume_lander = False,
    random_maps = True,
    save_interval = 10
)

gen.generate_graphs(num_graphs=10000)