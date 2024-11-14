import os
import numpy as np
import h5py
from planning_sandbox.environment.environment_class import Environment

current_directory = os.path.dirname(os.path.abspath(__file__))

num_agents = 3
num_goals = 5
num_skills = 2
size = 32
use_geo_data = True
random_map = True
assume_lander = False

num_samples = 1500000 


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
    map = env.grid_map.data

    observation_vector = np.concatenate((goals_map, agents_map, goal_required_skills, agent_skills), axis=0)
    return observation_vector, map

def generate_output(env: Environment):
    env.find_numerical_solution(solve_type='optimal')
    return np.array(env.get_action_vector())

# File to store data
data_filename = current_directory+'/datasets/dataset_3a_5g_random_maps.h5'

# Open HDF5 file in append mode
with h5py.File(data_filename, 'a') as h5f:

    # Initialize or get datasets
    if 'X' in h5f and 'y' in h5f and 'map' in h5f:
        X_dset = h5f['X']
        y_dset = h5f['y']
        map_dset = h5f['map']
    else:
        # Generate a sample to get the shape
        env = Environment(size=size, num_agents=num_agents, num_goals=num_goals,
                          num_skills=num_skills, use_geo_data=use_geo_data, random_map=random_map, assume_lander=assume_lander)
        input_array, map = generate_input(env)
        output_array = generate_output(env)
        input_shape = input_array.shape
        output_shape = output_array.shape
        map_shape = map.shape

        # Create datasets with maxshape set to None to allow resizing
        X_dset = h5f.create_dataset('X', shape=(0,) + input_shape,
                                    maxshape=(None,) + input_shape, chunks=True, compression="gzip")
        y_dset = h5f.create_dataset('y', shape=(0,) + output_shape,
                                    maxshape=(None,) + output_shape, chunks=True, compression="gzip")
        map_dset = h5f.create_dataset('map', shape=(0,) + map_shape,
                                    maxshape=(None,) + map_shape, chunks=True, compression="gzip")

    # Keep track of how many samples we already have
    num_existing_samples = X_dset.shape[0]

    env = Environment(size=size, num_agents=num_agents, num_goals=num_goals,
                      num_skills=num_skills, use_geo_data=use_geo_data, random_map=random_map, assume_lander=assume_lander)

    try:
        for i in range(num_samples):
            print(f"Generating sample {num_existing_samples + i + 1}", end='\r')
            env.reset()
            input_array, map_array = generate_input(env)
            output_array = generate_output(env)

            # Resize datasets to accommodate new data
            X_dset.resize((num_existing_samples + i + 1, ) + X_dset.shape[1:])
            y_dset.resize((num_existing_samples + i + 1, ) + y_dset.shape[1:])
            map_dset.resize((num_existing_samples + i + 1, ) + map_dset.shape[1:])

            # Add new data
            X_dset[num_existing_samples + i] = input_array
            y_dset[num_existing_samples + i] = output_array
            map_dset[num_existing_samples + i] = map_array

    except KeyboardInterrupt:
        print("\nData generation interrupted by user. Data saved to file.")

    else:
        print("\nData generation completed. Data saved to file.")