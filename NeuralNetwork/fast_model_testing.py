import os
import tensorflow as tf
import time
import keras
import numpy as np
import logging
import random
from planning_sandbox.environment_class import Environment
from planning_sandbox.visualizer_class import Visualizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# @keras.saving.register_keras_serializable()
# def custom_mae_with_rounded_zero_penalty(y_true, y_pred):
#     mae_loss = tf.reduce_mean(tf.abs(y_true - y_pred))
#     rounded_pred = tf.round(y_pred)
#     rounded_zero_mask = tf.cast(tf.equal(rounded_pred, 0), tf.float32)
#     non_zero_true_mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
#     penalty_mask = rounded_zero_mask * non_zero_true_mask
#     penalty = tf.reduce_sum(penalty_mask)
#     loss = mae_loss + 0.5 * penalty
#     return loss

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
vis = Visualizer(env, speed=50)


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
    solved = False
    pred_time = 0

    while not solved:
        input_array = generate_input(env)
        input_array = input_array.reshape(1, -1)
        pred_start = time.perf_counter_ns()
        predictions = model.predict(input_array)  # X_test is your test dataset inputs
        pred_time += time.perf_counter_ns() - pred_start
        predictions = tf.math.round(predictions[0])
        predictions = predictions.numpy().astype(int)
        full_solution = env.get_full_solution_from_action_vector(predictions)

        
        print(predictions)
        env.full_solution = full_solution
        vis.visualise_full_solution(soft_reset=False, fast=False)
        
        if env.scheduler.all_goals_claimed():
            solved = True
            print(f"Time taken: {pred_time/1e6} ms")
        else:   
            random.shuffle(env.agents)
            random.shuffle(env.goals)
            print("Deadlocked, resetting")
            print("Old prediction: ", predictions)