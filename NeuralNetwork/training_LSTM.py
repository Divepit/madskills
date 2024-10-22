import os
import tensorflow as tf
import keras
from keras import layers
import h5py
import numpy as np

from planning_sandbox.environment_class import Environment

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
current_directory = os.path.dirname(os.path.abspath(__file__))
dataset_path = current_directory + '/dataset.h5'

env = Environment(size=100, num_agents=3, num_goals=5, num_skills=2, use_geo_data=True)

with h5py.File(dataset_path, 'r') as h5f:
    X = h5f['X'][:]
    y = h5f['y'][:]

new_y = []
for action_vector in y:
    new_y.append(env.get_sequential_action_vectors(action_vector))
y = np.array(new_y)


n_agents = y[0].shape[1]
n_timesteps = y[0].shape[0]

# new_X = []
# for observation_vector in X:
#     goal_positions = []
#     agent_positions = []
#     goal_required_skills = []
#     agent_skills = []
#     for i in range(0, len(env.goals), 2):
#         goal_positions.append([observation_vector[i], observation_vector[i+1]])
#     for i in range(len(env.goals)*2, len(env.goals)*2 + len(env.agents)*2, 2):
#         agent_positions.append(np.array([observation_vector[i], observation_vector[i+1]]))
#     for i in range(len(env.goals)*2 + len(env.agents)*2, len(env.goals)*2 + len(env.agents)*2 + len(env.goals)*env.num_skills):
#         goal_required_skills.append(observation_vector[i])
#     for i in range(len(env.goals)*2 + len(env.agents)*2 + len(env.goals)*env.num_skills, len(env.goals)*2 + len(env.agents)*2 + len(env.goals)*env.num_skills + len(env.agents)*env.num_skills):
#         agent_skills.append(observation_vector[i])
#     new_X.append([np.array(goal_positions), np.array(agent_positions), np.array(goal_required_skills), np.array(agent_skills)])

# X = new_X

print(X[0].shape)

print(f"Number of agents: {n_agents}")
print(f"Number of timesteps: {n_timesteps}")


model = keras.Sequential()
model.add(keras.Input(shape=X[0].shape))
model.add(layers.LSTM(128, activation='relu', return_sequences=True, input_shape=X[0].shape))


model.build(input_shape=(None, len(X[0])))
print(model.summary())

losses = [keras.losses.MeanAbsoluteError() for _ in range(n_timesteps)]
optim = keras.optimizers.Adam(learning_rate=0.0001)
metrics = ['accuracy' for _ in range(n_timesteps)]

model.compile(loss=losses, optimizer=optim, metrics=metrics)

batch_size = 32
epochs = 100

# Prepare the data
X = [np.array([x[i] for x in X]) for i in range(4)]  # 4 inputs
y = [np.array([y_sample[t] for y_sample in y]) for t in range(n_timesteps)]  # n_timesteps outputs

# Call model.fit
# model.fit(X, y, batch_size=batch_size, epochs=epochs)
# model.fit(X, [y[:, i] for i in range(n_timesteps)], epochs=epochs, batch_size=batch_size, shuffle=True, verbose=2)




# loss = keras.losses.MeanAbsoluteError()
# optim = keras.optimizers.Adam(learning_rate=0.01)
# metrics = ['accuracy']

# model.compile(loss=loss, optimizer=optim, metrics=metrics)

# batch_size = 32
# epochs = 100

# model.fit(X, y, epochs=epochs, batch_size=batch_size, shuffle=True, verbose=1)

# model.evaluate(X, y, batch_size=batch_size, verbose=1)