import os
from madskills.machine_learning.mlp_solver.dataset_generator_class import DatasetGenerator

current_directory = os.path.dirname(os.path.abspath(__file__))
dataset_path = current_directory + '/datasets/tester.h5'

gen = DatasetGenerator(
    save_path=dataset_path,
    num_agents=3,
    num_goals=5,
    num_skills=2,
    size=32,
    use_geo_data=True,
    random_map=True,
    assume_lander=False,
    overwrite_protection=True
)

gen.generate_data(20000)
