import os
import glob
import logging
import copy
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import A2C

from reinforcement_learning.RL_env import ILEnv
from planning_sandbox.environment.visualizer_class import Visualizer

final_num_agents = 3
final_num_goals = 5
num_skills = 2
final_size = 32
advances = 0

save_path = "/Users/marco/Programming/PlanningEnvironmentLibrary/ImitationLearning/model_logs/"
learningEnv = ILEnv(final_num_agents=final_num_agents, final_num_goals=final_num_goals, final_size=final_size, num_skills=num_skills)
# learningEnv.advance = True
for _ in range(advances):
    learningEnv.advance = True
    learningEnv.reset()

while True:
    learningEnv.reset()
    dispEnv = copy.deepcopy(learningEnv.sandboxEnv)
    # optimalEnv = copy.deepcopy(learningEnv.sandboxEnv)
    # optimalEnv.find_numerical_solution(solve_type='optimal')
    # optimalEnv.solve_full_solution()
    # optimal_solution_cost = optimalEnv.get_agent_benchmarks()[2]
    vis = Visualizer(dispEnv, speed=20)


    files = glob.glob(os.path.join(save_path, "*.zip"))
    latest_file = max(files, key=os.path.getmtime)
    model = A2C.load(latest_file)

    print(f"Loading model from {latest_file}")

    while not dispEnv.scheduler.all_goals_claimed():
        print("Predicting action")
        obs = dispEnv.get_observation_vector(
            pad_agents=learningEnv.final_num_agents - learningEnv.current_num_agents, 
            pad_goals=learningEnv.final_num_goals - learningEnv.current_num_goals, 
            pad_map=learningEnv.final_size ** 2 - learningEnv.sandboxEnv.grid_map.downscaled_data.shape[0] ** 2
        )
        action, _ = model.predict(obs, deterministic=False)
        dispEnv.full_solution = dispEnv.get_full_solution_from_action_vector(action)
        vis.visualise_full_solution(soft_reset=False)

    predicted_full_solution_cost = dispEnv.get_agent_benchmarks()[2]
    
    del vis
    del dispEnv

    # print(f"Optimal solution cost: {optimal_solution_cost}")
    print(f"Predicted solution cost: {predicted_full_solution_cost}")


