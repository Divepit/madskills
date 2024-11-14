# %% Test Trained Network
import os
from planning_sandbox.machine_learning.mlp_solver.mlp_evaluator_class import MlpEvaluator
from planning_sandbox.environment.environment_class import Environment

current_directory = os.path.dirname(os.path.abspath(__file__))
autoencoder_path = current_directory + '/models/autoencoder.keras'
model_save_path = current_directory + '/models/mlp.keras'


num_agents = 3
num_goals = 5
num_skills = 2
size = 64
use_geo_data = True
random_map = True
assume_lander = False

# Create environment (needed for input generation and visualization)
env = Environment(size=size, num_agents=num_agents, num_goals=num_goals,
                  num_skills=num_skills, use_geo_data=use_geo_data, random_map=random_map, assume_lander=assume_lander)

evaluator = MlpEvaluator(model_path=model_save_path, autoencoder_path=autoencoder_path, env=env)
evaluator.perform_visual_evaluation()
evaluator.perform_comparison_to_optimal_solver(runs=10)