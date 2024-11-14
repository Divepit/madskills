from skimage.transform import resize
import tensorflow as tf
import numpy as np
import keras
from planning_sandbox.environment.environment_class import Environment

def downscale_data(data, map_size, mpp=5):

    current_height, current_width = data.shape
    current_pixel_size = mpp * max(current_height, current_width) / max(map_size, map_size)

    scale_factor = current_pixel_size / mpp

    new_height = int(current_height / scale_factor)
    new_width = int(current_width / scale_factor)

    downscaled_data = resize(data, (new_height, new_width), order=1, mode='reflect', anti_aliasing=True)

    return downscaled_data

@keras.saving.register_keras_serializable()
def _rounded_accuracy(y_true, y_pred):
    y_pred_rounded = tf.round(y_pred)
    correct_predictions = tf.cast(tf.equal(y_true, y_pred_rounded), dtype=tf.float32)
    return tf.reduce_mean(correct_predictions)

def generate_mlp_input_from_env(env: Environment):
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