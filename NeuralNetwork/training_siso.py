import os
import keras
import h5py
from matplotlib import pyplot as plt


from planning_sandbox.environment_class import Environment

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
current_directory = os.path.dirname(os.path.abspath(__file__))
dataset_path = current_directory + '/dataset.h5'

env = Environment(size=100, num_agents=3, num_goals=5, num_skills=2, use_geo_data=True)

with h5py.File(dataset_path, 'r') as h5f:
    X = h5f['X'][:]
    y = h5f['y'][:]



model = keras.Sequential(
    [
        keras.Input(shape=(X[0].shape)),
        keras.layers.Dense(1024, activation='relu'),
        keras.layers.Dense(1024, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(1024, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(1024, activation='relu'),
        keras.layers.Dense(1024, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(1024, activation='relu'),
        keras.layers.Dense(1024, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(1024, activation='relu'),
        keras.layers.Dense(1024, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(1024, activation='relu'),
        # keras.layers.Dense(512, activation='relu'),
        # keras.layers.Dense(512, activation='relu'),
        # keras.layers.Dropout(0.3),
        # keras.layers.Dense(256, activation='relu'),
        # keras.layers.Dense(256, activation='relu'),
        # keras.layers.Dense(256, activation='relu'),
        # keras.layers.Dropout(0.3),
        # keras.layers.Dense(256, activation='relu'),
        # keras.layers.Dense(256, activation='relu'),
        keras.layers.Dense(len(y[0]), activation='linear'),
    ]
)

early_stopping = keras.callbacks.EarlyStopping(
    patience=10,
    restore_best_weights=True,
)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.1,
    patience=5,
    min_lr=0.00001,
)

loss = keras.losses.MeanAbsoluteError()
optim = keras.optimizers.Adam(learning_rate=0.001)
metrics = ['mae']

model.compile(loss=loss, optimizer=optim, metrics=metrics)

batch_size = 600
epochs = 1000

history = model.fit(X, y, epochs=epochs, batch_size=batch_size, shuffle=True, verbose=1, callbacks=[early_stopping, reduce_lr], validation_split=0.3)

model.evaluate(X, y, batch_size=batch_size, verbose=1)

model.save(current_directory+'/model.keras')

# Plot loss and accuracy
plt.figure(figsize=(12, 5))

# Plot training & validation loss
plt.subplot(1, 1, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()


plt.tight_layout()
plt.show()