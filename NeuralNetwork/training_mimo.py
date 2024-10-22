import os
import tensorflow as tf
import keras
from keras import layers
import h5py
import numpy as np

from planning_sandbox.environment_class import Environment

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
current_directory = os.path.dirname(os.path.abspath(__file__))
dataset_path = current_directory + '/dataset_simple.h5'

env = Environment(size=32, num_agents=1, num_goals=3, num_skills=1, use_geo_data=False)

with h5py.File(dataset_path, 'r') as h5f:
    X = h5f['X'][:]
    y = h5f['y'][:]

new_y = []
for action_vector in y:
    new_y.append(env.get_sequential_action_vectors(action_vector))
y = np.array(new_y)


n_agents = y[0].shape[1]
n_timesteps = y[0].shape[0]

new_X = []
for observation_vector in X:
    goal_positions = []
    agent_positions = []
    goal_required_skills = []
    agent_skills = []
    for i in range(0, len(env.goals), 2):
        goal_positions.append([observation_vector[i], observation_vector[i+1]])
    for i in range(len(env.goals)*2, len(env.goals)*2 + len(env.agents)*2, 2):
        agent_positions.append(np.array([observation_vector[i], observation_vector[i+1]]))
    for i in range(len(env.goals)*2 + len(env.agents)*2, len(env.goals)*2 + len(env.agents)*2 + len(env.goals)*env.num_skills):
        goal_required_skills.append(observation_vector[i])
    for i in range(len(env.goals)*2 + len(env.agents)*2 + len(env.goals)*env.num_skills, len(env.goals)*2 + len(env.agents)*2 + len(env.goals)*env.num_skills + len(env.agents)*env.num_skills):
        agent_skills.append(observation_vector[i])
    new_X.append([np.array(goal_positions), np.array(agent_positions), np.array(goal_required_skills), np.array(agent_skills)])

X = new_X

print(X[0])

print(f"Number of agents: {n_agents}")
print(f"Number of timesteps: {n_timesteps}")

inputs1 = keras.Input(shape=(X[0][0].shape))
inputs2 = keras.Input(shape=(X[0][1].shape))
inputs3 = keras.Input(shape=(X[0][2].shape))
inputs4 = keras.Input(shape=(X[0][3].shape))


flatten = keras.layers.Flatten()

dense1 = layers.Dense(512, activation='relu')
dense2 = layers.Dense(512, activation='relu')
dense3 = layers.Dense(512, activation='relu')
dense4 = layers.Dense(512, activation='relu')
dense5 = layers.Dense(512, activation='relu')
dense6 = layers.Dense(512, activation='relu')

for i in range(n_timesteps):
    globals()[f'timestep_{i+1}'] = layers.Dense(n_agents, activation='relu', name=f'timestep_{i+1}')
    

x1 = flatten(inputs1)


x2 = flatten(inputs2)


x3 = flatten(inputs3)


x4 = flatten(inputs4)




x = layers.concatenate([x1, x2, x3, x4])
x = dense1(x)
x = dense2(x)
x = dense3(x)
x = dense4(x)
x = dense5(x)
x = dense6(x)


for i in range(n_timesteps):
    globals()[f'outputs_{i+1}'] = globals()[f'timestep_{i+1}'](x)



model = keras.Model(inputs=[inputs1,inputs2,inputs3,inputs4], outputs=[globals()[f'outputs_{i+1}'] for i in range(n_timesteps)])
model.build(input_shape=(None, len(X[0])))
print(model.summary())

losses = [keras.losses.MeanAbsoluteError() for _ in range(n_timesteps)]
optim = keras.optimizers.Adam(learning_rate=0.0001)
metrics = ['accuracy' for _ in range(n_timesteps)]

model.compile(loss=losses, optimizer=optim, metrics=metrics)

batch_size = 1
epochs = 100

# Prepare the data
X = [np.array([x[i] for x in X]) for i in range(4)]  # 4 inputs
y = [np.array([y_sample[t] for y_sample in y]) for t in range(n_timesteps)]  # n_timesteps outputs

# Call model.fit
model.fit(X, y, batch_size=batch_size, epochs=epochs)
# model.fit(X, [y[:, i] for i in range(n_timesteps)], epochs=epochs, batch_size=batch_size, shuffle=True, verbose=2)




# loss = keras.losses.MeanAbsoluteError()
# optim = keras.optimizers.Adam(learning_rate=0.01)
# metrics = ['accuracy']

# model.compile(loss=loss, optimizer=optim, metrics=metrics)

# batch_size = 32
# epochs = 100

# model.fit(X, y, epochs=epochs, batch_size=batch_size, shuffle=True, verbose=1)

# model.evaluate(X, y, batch_size=batch_size, verbose=1)