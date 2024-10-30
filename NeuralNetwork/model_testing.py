import os
import random
import tensorflow as tf
import time
import keras
import numpy as np
import logging
import matplotlib.pyplot as plt  # Added for plotting
from planning_sandbox.environment_class import Environment
from planning_sandbox.visualizer_class import Visualizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@keras.saving.register_keras_serializable()
def custom_mae_with_rounded_zero_penalty(y_true, y_pred):
    mae_loss = tf.reduce_mean(tf.abs(y_true - y_pred))
    rounded_pred = tf.round(y_pred)
    rounded_zero_mask = tf.cast(tf.equal(rounded_pred, 0), tf.float32)
    non_zero_true_mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
    penalty_mask = rounded_zero_mask * non_zero_true_mask
    penalty = tf.reduce_sum(penalty_mask)
    loss = mae_loss + 0.5 * penalty
    return loss

@keras.saving.register_keras_serializable()
def rounded_accuracy(y_true, y_pred):
    # Round the predictions to the nearest integer
    y_pred_rounded = tf.round(y_pred)
    
    # Check if the rounded predictions match the true values
    correct_predictions = tf.cast(tf.equal(y_true, y_pred_rounded), dtype=tf.float32)
        
    # Calculate the mean accuracy
    return tf.reduce_mean(correct_predictions)

current_directory = os.path.dirname(os.path.abspath(__file__))
model = keras.models.load_model(current_directory+'/models/good_3a_5g_2sk_100x100_27min.keras')
# model = keras.models.load_model(current_directory+'/model.keras')

# Parameters (should match data generation parameters)
num_agents = 3
num_goals = 5
num_skills = 2
size = 200
use_geo_data = True

# Create environment (needed for input generation and visualization)
env = Environment(size=size, num_agents=num_agents, num_goals=num_goals,
                  num_skills=num_skills, use_geo_data=use_geo_data)


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


runs = 200
successes = []
optimal_costs = []
predicted_costs = []
optimal_times = []
predicted_times = []

pred_time = 0
pred_cost = 0
opt_cost = 0

for i in range(runs):
    print(f"Run {i+1}/{runs}")
    env.reset()
    solved = False
    optmial_found = False
    while not solved:
        # print("Input Array:", input_array)

        if not optmial_found:
            optmial_found = True
            opt_start = time.perf_counter_ns()
            optimal_solution, new_opt_cost = env.find_numerical_solution(solve_type='optimal')
            opt_end = time.perf_counter_ns()
            env.solve_full_solution(fast=False, soft_reset=True)
            opt_cost = env.get_agent_benchmarks()[2]
            env.soft_reset()

        input_array = generate_input(env)
        input_array = input_array.reshape(1, -1)
        pred_start = time.perf_counter_ns()
        predictions = model.predict(input_array, verbose=0)  
        pred_end = time.perf_counter_ns()
        pred_time += (pred_end - pred_start) / 1e6
        predictions = tf.math.round(predictions[0])
        predictions = predictions.numpy().astype(int)
        full_solution = env.get_full_solution_from_action_vector(predictions)
        env.full_solution = full_solution
        env.solve_full_solution(fast=False, soft_reset=False)
        pred_cost = env.get_agent_benchmarks()[2]
        
        if env.scheduler.all_goals_claimed():
            solved = True
            successes.append(1)
            optimal_costs.append(opt_cost)
            predicted_costs.append(pred_cost)
            optimal_times.append((opt_end - opt_start) / 1e6)
            predicted_times.append(pred_time)
            print(" ====== RESULTS ====== ")
            print(f"Optimal Cost: {opt_cost}")
            print(f"Predicted Cost: {pred_cost}")
            print(f"Cost Difference: {pred_cost-opt_cost}")
            print(f"Optimal Time: {(opt_end - opt_start) / 1e6} ms")
            print(f"Predicted Time: {pred_time} ms")
            print(" ===================== ")
            opt_cost = 0
            pred_time = 0
            pred_cost = 0
        else:
            random.shuffle(env.goals)

print(f"Success Rate: {sum(successes)/len(successes)}")

# Compare costs and times of optimal and predicted solutions
optimal_costs = np.array(optimal_costs)
predicted_costs = np.array(predicted_costs)
cost_diff = predicted_costs - optimal_costs

optimal_times = np.array(optimal_times)
predicted_times = np.array(predicted_times)
time_diff = predicted_times - optimal_times

print(f"Mean Optimal Cost: {np.mean(optimal_costs)}")
print(f"Mean Predicted Cost: {np.mean(predicted_costs)}")
print(f"Mean Optimal Time: {np.mean(optimal_times)} ms")
print(f"Mean Predicted Time: {np.mean(predicted_times)} ms")

print(f"Mean Cost Difference: {np.mean(cost_diff)}")
print(f"Max Cost Difference: {np.max(cost_diff)}")
print(f"Min Cost Difference: {np.min(cost_diff)}")
print(f"Cost Difference Standard Deviation: {np.std(cost_diff)}")
print(f"Cost Difference Variance: {np.var(cost_diff)}")
print(f"Cost Difference Median: {np.median(cost_diff)}")

# ===========================
# Plotting code starts here
# ===========================

import matplotlib.pyplot as plt  # Already imported at the top

# Plot settings for better visuals
plt.style.use('seaborn-v0_8-darkgrid')  # Use an available style
plt.rcParams.update({'figure.figsize': (10, 6), 'figure.dpi': 100})

# Include environment parameters for reference
env_params = f'Agents: {num_agents}, Goals: {num_goals}, Skills: {num_skills}, Size: {size}, Runs: {runs}'

# 1. Bar Chart of Mean Computation Times
plt.figure()
times = [np.mean(optimal_times), np.mean(predicted_times)]
labels = ['Expert Solution', 'NN Model']
plt.bar(labels, times, color=['blue', 'green'])
plt.title('Mean Computation Times')
plt.ylabel('Time (ms)')
plt.text(0.5, max(times)*0.9, env_params, ha='center', fontsize=9)
plt.savefig('mean_computation_times.png')
plt.close()
print("Saved 'mean_computation_times.png'.")

# 2. Boxplot of Cost Differences
plt.figure()
plt.boxplot(cost_diff, vert=False)
plt.title('Cost Difference Between NN Model and Expert Solution')
plt.xlabel('Cost Difference (NN Model Cost - Expert Cost)')
plt.text(np.mean(cost_diff), 1.05, env_params, ha='center', fontsize=9)
plt.savefig('cost_difference_boxplot.png')
plt.close()
print("Saved 'cost_difference_boxplot.png'.")

# 3. Bar Chart of Mean Costs
plt.figure()
costs = [np.mean(optimal_costs), np.mean(predicted_costs)]
labels = ['Expert Solution', 'NN Model']
plt.bar(labels, costs, color=['blue', 'green'])
plt.title('Mean Costs')
plt.ylabel('Total Cost (units)')  # Replace 'units' with actual units if known
plt.text(0.5, max(costs)*0.9, env_params, ha='center', fontsize=9)
plt.savefig('mean_costs.png')
plt.close()
print("Saved 'mean_costs.png'.")

# 4. Scatter Plot of Computation Time Difference vs. Cost Difference
plt.figure()
plt.scatter(cost_diff, time_diff, color='purple', alpha=0.7)
plt.title('Computation Time Difference vs. Cost Difference (NN Model Time - Expert Time)')
plt.xlabel('Cost Difference (NN Model Cost - Expert Cost)')
plt.ylabel('Computation Time Difference (ms)')
plt.axhline(0, color='grey', linestyle='--', linewidth=1)
plt.axvline(0, color='grey', linestyle='--', linewidth=1)
plt.text(0.5, max(time_diff)*0.9, env_params, ha='center', fontsize=9)
plt.savefig('time_vs_cost_difference.png')
plt.close()
print("Saved 'time_vs_cost_difference.png'.")

# ===========================
# Plotting code ends here
# ===========================