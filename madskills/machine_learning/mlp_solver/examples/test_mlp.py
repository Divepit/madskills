# %% Test Trained Network
import os
import time
from madskills.machine_learning.mlp_solver.mlp_evaluator_class import MlpEvaluator
from madskills.environment.environment_class import Environment

current_directory = os.path.dirname(os.path.abspath(__file__))
autoencoder_path = None #current_directory + '/models/autoencoder.keras'
model_save_path = current_directory + '/models/3a_5g_2sk_32x32_fixed_map.keras'
plots_save_path = current_directory + '/plots/'+time.strftime("%Y%m%d-%H%M%S")


num_agents = 3
num_goals = 5
num_skills = 2
size = 200
use_geo_data = True
random_map = False
assume_lander = True

# Create environment (needed for input generation and visualization)
env = Environment(size=size, num_agents=num_agents, num_goals=num_goals,
                  num_skills=num_skills, use_geo_data=use_geo_data, random_map=random_map, assume_lander=assume_lander)

evaluator = MlpEvaluator(model_path=model_save_path, autoencoder_path=autoencoder_path, env=env, plots_directory=plots_save_path)
# evaluator.perform_visual_evaluation()
evaluator.perform_comparison_to_optimal_solver(runs=20, visualize=True)