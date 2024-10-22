import os
import tensorflow as tf
import time
import keras
import numpy as np
import logging
from planning_sandbox.environment_class import Environment
from planning_sandbox.visualizer_class import Visualizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

current_directory = os.path.dirname(os.path.abspath(__file__))
# model = keras.models.load_model(current_directory+'/models/75p_2a_3g_2sk_32x32.keras')
model = keras.models.load_model(current_directory+'/model.keras')

# Parameters (should match data generation parameters)
num_agents = 3
num_goals = 5
num_skills = 2
size = 100
use_geo_data = True

# Create environment (needed for input generation and visualization)
env = Environment(size=size, num_agents=num_agents, num_goals=num_goals,
                  num_skills=num_skills, use_geo_data=use_geo_data)
vis = Visualizer(env, speed=20)

def generate_input(env: Environment):
    goals_map = []
    for goal in env.goals:
        goals_map.append(goal.position[0]/env.size)
        goals_map.append(goal.position[1]/env.size)
    goals_map = np.array(goals_map, dtype=np.float32)
    agents_map = []
    for agent in env.agents:
        agents_map.append(agent.position[0]/env.size)
        agents_map.append(agent.position[1]/env.size)
    agents_map = np.array(agents_map, dtype=np.float32)

    goal_required_skills = np.array([[(1 if skill in goal.required_skills else 0) for skill in range(env.num_skills)] for goal in env.goals], dtype=np.float32).flatten()
    agent_skills = np.array([[(1 if skill in agent.skills else 0) for skill in range(env.num_skills)] for agent in env.agents], dtype=np.float32).flatten()

    observation_vector = np.concatenate((goals_map, agents_map, goal_required_skills, agent_skills))
    return observation_vector



while True:
    env.reset()
    input_array = generate_input(env)
    # print("Input Array:", input_array)
    input_array = input_array.reshape(1, -1)

    opt_start = time.perf_counter_ns()
    optimal_solution, opt_cost = env.find_numerical_solution(solve_type='optimal')
    opt_end = time.perf_counter_ns()

    # opt_cost = env.solve_full_solution(fast=True)[2]

    env.soft_reset()

    # output_array = model.predict(input_array)
    # print("Output Array:", output_array)
    # Assume 'predictions' is a list of arrays, one for each agent, returned by model.predict()
    pred_start = time.perf_counter_ns()
    predictions = model.predict(input_array)  # X_test is your test dataset inputs
    pred_end = time.perf_counter_ns()
    predictions = tf.math.round(predictions[0])
    predictions = predictions.numpy().astype(int)


    full_solution = env.get_full_solution_from_action_vector(predictions)
    pred_cost = env.calculate_cost_of_closed_solution(full_solution)

    

    # print(f"PREDICTIONS: {predictions}")

    # # For each agent, get the predicted goal indices
    # agent_predictions_list = []
    # for agent in range(num_agents):
    #     # predictions[agent] has shape (num_samples, time_steps, num_goals)
    #     agent_probs = predictions[agent]
    #     # Apply argmax to get predicted goal indices
    #     agent_preds = np.argmax(agent_probs, axis=-1)  # Shape: (num_samples, time_steps)
    #     agent_predictions_list.append(agent_preds)

    # print(agent_predictions_list)

    # solution = {}
    # for agent_index, agent in enumerate(env.agents):
    #     goal_list = [env.goals[goal_index-1] for goal_index in agent_predictions_list[agent_index][0] if goal_index != 0]
    #     solution[agent] = goal_list

    env.full_solution = full_solution
    vis.visualise_full_solution()
    
    if not env.deadlocked:
        print(" ====== RESULTS ====== ")
        print(f"Optimal Cost: {opt_cost}")
        print(f"Predicted Cost: {pred_cost}")
        print(f"Optimal Time: {(opt_end - opt_start) / 1e6} ms")
        print(f"Predicted Time: {(pred_end - pred_start) / 1e6} ms")
        print(" ===================== ")

    # predicted_output_int = np.rint(output_array).astype(int)

    # print('Predicted Integer Output:', predicted_output_int)