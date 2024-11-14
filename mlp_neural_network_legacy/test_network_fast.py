import os
import tensorflow as tf
import time
import keras
import numpy as np
import logging
import random
from planning_sandbox.environment.environment_class import Environment
from planning_sandbox.environment.visualizer_class import Visualizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@keras.saving.register_keras_serializable()
class Autoencoder(keras.Model):
    def __init__(self, latent_dim, shape):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.shape = shape
        self.encoder = keras.Sequential([
            keras.layers.Flatten(),
            keras.layers.Dense(256),
            keras.layers.LeakyReLU(alpha=0.1),
            keras.layers.Dense(128),
            keras.layers.LeakyReLU(alpha=0.1),
            keras.layers.Dense(latent_dim),
            keras.layers.LeakyReLU(alpha=0.1)
        ])
        self.decoder = keras.Sequential([
            keras.layers.Dense(256),
            keras.layers.LeakyReLU(alpha=0.1),
            keras.layers.Dense(tf.math.reduce_prod(shape).numpy()),  # No activation here
            keras.layers.Reshape(shape)
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def get_config(self):
        # Return the configuration of the model to enable deserialization
        return {
            'latent_dim': self.latent_dim,
            'shape': self.shape
        }

    @classmethod
    def from_config(cls, config):
        # Create a new instance from the config dictionary
        return cls(**config)

@keras.saving.register_keras_serializable()
def rounded_accuracy(y_true, y_pred):
    y_pred_rounded = tf.round(y_pred)
    correct_predictions = tf.cast(tf.equal(y_true, y_pred_rounded), dtype=tf.float32)
    return tf.reduce_mean(correct_predictions)

current_directory = os.path.dirname(os.path.abspath(__file__))
model = keras.models.load_model(current_directory+'/model.keras')
autoencoder = keras.models.load_model(current_directory+'/autoencoder.keras')
# model = keras.models.load_model(current_directory+'/model.keras')

num_agents = 3
num_goals = 5
num_skills = 2
size = 64
use_geo_data = True
random_map = True
assume_lander = False

env = Environment(size=size, num_agents=num_agents, num_goals=num_goals,
                  num_skills=num_skills, use_geo_data=use_geo_data, random_map=random_map, assume_lander=assume_lander)
vis = Visualizer(env, speed=50)

def encode_map(env: Environment):
    return autoencoder.encoder(np.array([env.grid_map.downscaled_data]).astype(np.float32))[0]


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
    optmial_found = False
    map = encode_map(env)
    while not solved:

        input_array = generate_input(env)
        input_array = np.concatenate((input_array, map))
        input_array = input_array.reshape(1, -1)
        predictions = model.predict(input_array, verbose=0)  
        predictions = tf.math.round(predictions[0])
        predictions = predictions.numpy().astype(int)
        full_solution = env.get_full_solution_from_action_vector(predictions)
        env.full_solution = full_solution
        # env.solve_full_solution(fast=False, soft_reset=False)
        vis.visualise_full_solution(soft_reset=False)
        pred_cost = env.get_agent_benchmarks()[2]
        
        if env.scheduler.all_goals_claimed():
            solved = True
        else:
            random.shuffle(env.goals)

# while True:
#     env.reset()
#     solved = False
#     pred_time = 0

#     while not solved:
#         input_array = generate_input(env)
#         input_array = input_array.reshape(1, -1)
#         pred_start = time.perf_counter_ns()
#         predictions = model.predict(input_array)  # X_test is your test dataset inputs
#         pred_time += time.perf_counter_ns() - pred_start
#         predictions = tf.math.round(predictions[0])
#         predictions = predictions.numpy().astype(int)
#         full_solution = env.get_full_solution_from_action_vector(predictions)

        
#         print(predictions)
#         env.full_solution = full_solution
#         vis.visualise_full_solution(soft_reset=False, fast=False)
        
#         if env.scheduler.all_goals_claimed():
#             solved = True
#             print(f"Time taken: {pred_time/1e6} ms")
#         else:   
#             random.shuffle(env.agents)
#             random.shuffle(env.goals)
#             print("Deadlocked, resetting")
#             print("Old prediction: ", predictions)