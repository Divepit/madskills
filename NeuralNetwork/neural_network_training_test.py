import os
import numpy as np
from sklearn.model_selection import train_test_split
import keras
import matplotlib.pyplot as plt
import h5py



from planning_sandbox.environment_class import Environment
from planning_sandbox.visualizer_class import Visualizer


current_directory = os.path.dirname(os.path.abspath(__file__))

# Placeholder function for data generation
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

# File to read data from
data_filename = current_directory+'/dataset.h5'








# Parameters (should match data generation parameters)
num_agents = 3
num_goals = 5
num_skills = 2
size = 100
use_geo_data = True

# Create environment (needed for input generation and visualization)
env = Environment(size=size, num_agents=num_agents, num_goals=num_goals,
                  num_skills=num_skills, use_geo_data=use_geo_data)
# Read data from HDF5 file
with h5py.File(data_filename, 'r') as h5f:
    X = h5f['X'][:]
    y = h5f['y'][:]
new_y = []
for action_vector in y:
    new_y.append(env.get_sequential_action_vectors(action_vector))
y = np.array(new_y)
print(f'Amount of data: {X.shape[0]} samples')


# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42)

input_length = X_train.shape[1]
time_steps = y_train.shape[1]

print(input_length)

inputs = keras.layers.Input(shape=(input_length,))
# Dense layers to process the input
x = keras.layers.Dense(128, activation='relu')(inputs)
x = keras.layers.Dense(128, activation='relu')(x)
# Repeat the vector to match the number of time steps
x = keras.layers.RepeatVector(time_steps)(x)

# LSTM layer
x = keras.layers.LSTM(128, return_sequences=True)(x)

# Outputs for each agent
outputs = []
for agent in range(num_agents):
    # TimeDistributed layer for each agent
    agent_output = keras.layers.TimeDistributed(keras.layers.Dense(num_goals+1, activation='softmax'), name=f'agent_{agent+1}_output')(x)
    outputs.append(agent_output)

# Define the model
model = keras.models.Model(inputs=inputs, outputs=outputs)

# neurons = 300

# # Define the model architecture
# model = keras.Sequential([
#     keras.layers.Dense(neurons, activation='relu', input_shape=(input_length,)),
#     keras.layers.Dense(neurons, activation='relu'),
#     keras.layers.Dense(neurons, activation='relu'),
#     keras.layers.Dense(neurons, activation='relu'),
#     keras.layers.Dense(neurons, activation='relu'),
#     keras.layers.Dense(output_length)
# ])

# Compile the model
# model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

losses = {f'agent_{agent+1}_output': 'sparse_categorical_crossentropy' for agent in range(num_agents)}
metrics = {f'agent_{agent+1}_output': 'accuracy' for agent in range(num_agents)}

model.compile(optimizer='adam', loss=losses, metrics=metrics)

# Prepare target dictionaries for each agent
y_train_dict = {f'agent_{agent+1}_output': y_train[:, :, agent] for agent in range(num_agents)}
y_val_dict = {f'agent_{agent+1}_output': y_val[:, :, agent] for agent in range(num_agents)}

# Set up early stopping
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=200,
    restore_best_weights=True
)

# Train the model
history = model.fit(
    X_train, y_train_dict,
    epochs=200,
    batch_size=16,
    validation_data=(X_val, y_val_dict),
    callbacks=[early_stopping]
)

# # Set up early stopping to prevent overfitting
# early_stopping = keras.callbacks.EarlyStopping(
#     monitor='val_loss',
#     patience=5,
#     restore_best_weights=True
# )

# # Train the model
# history = model.fit(
#     X_train, y_train,
#     epochs=200,  # Adjust the number of epochs as needed
#     batch_size=16,
#     validation_data=(X_val, y_val),
#     callbacks=[early_stopping]
# )

model.save(current_directory+'/model.keras')

# Evaluation
val_loss, val_mae = model.evaluate(X_val, y_val)
print(f'Validation Loss: {val_loss}')
print(f'Validation MAE: {val_mae}')

# Visualization
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Train')
plt.plot(history.history['val_mae'], label='Validation')
plt.title('Model MAE')
plt.xlabel('Epoch')
plt.ylabel('Mean Absolute Error')
plt.legend()

plt.show(block=False)

# Use the model to make a prediction

while True:
    env.reset()
    input_array = generate_input(env)
    print("Input Array:", input_array)
    input_array = input_array.reshape(1, -1)

    output_array = model.predict(input_array)[0]
    print("Output Array:", output_array)
    predicted_output_int = np.rint(output_array).astype(int)

    print('Predicted Integer Output:', predicted_output_int)

    # # Use the predicted output to get the full solution
    # optimal_solution, optimal_cost = env.find_numerical_solution(solve_type='optimal')

    # predicted_solution = env.get_full_solution_from_action_vector(predicted_output_int)
    # env.full_solution = predicted_solution

    # predicted_cost = env.calculate_cost_of_closed_solution(predicted_solution)

    # print(f'Predicted Cost: {predicted_cost}')
    # print(f'Optimal Cost: {optimal_cost}')

    # # Visualize the solution
    # vis = Visualizer(env, speed=100)
    # vis.visualise_full_solution()

