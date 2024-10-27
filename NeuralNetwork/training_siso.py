import os
import keras
import h5py
import tensorflow as tf
from matplotlib import pyplot as plt
from datetime import datetime

from planning_sandbox.environment_class import Environment

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
current_directory = os.path.dirname(os.path.abspath(__file__))
dataset_path = current_directory + '/dataset_3a_5g_2sk_100x100_map.h5'

with h5py.File(dataset_path, 'r') as h5f:
    X = h5f['X'][:]
    y = h5f['y'][:]

nodes = 2048
slope = 0.1
dropout = 0.1

model = keras.Sequential(
    [
        keras.Input(shape=(X[0].shape)),
        keras.layers.Dense(nodes),
        keras.layers.LeakyReLU(slope),  # Applying LeakyReLU with a small negative slope (alpha)
        keras.layers.Dense(nodes),
        keras.layers.LeakyReLU(slope),
        keras.layers.Dropout(dropout),
        keras.layers.Dense(nodes),
        keras.layers.LeakyReLU(slope),
        keras.layers.Dense(nodes),
        keras.layers.LeakyReLU(slope),
        keras.layers.Dropout(dropout),
        keras.layers.Dense(nodes),
        keras.layers.LeakyReLU(slope),
        keras.layers.Dense(nodes),
        keras.layers.LeakyReLU(slope),
        keras.layers.Dropout(dropout),
        keras.layers.Dense(nodes),
        keras.layers.LeakyReLU(slope),
        keras.layers.Dense(nodes),
        keras.layers.LeakyReLU(slope),
        keras.layers.Dense(nodes),
        keras.layers.LeakyReLU(slope),
        keras.layers.Dense(len(y[0])),
        keras.layers.LeakyReLU(slope),
    ]
)

early_stopping = keras.callbacks.EarlyStopping(
    patience=5,
    restore_best_weights=True,
)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.1,
    patience=3,
    min_lr=0.000015,
)

tensorboard_callback = keras.callbacks.TensorBoard(log_dir=current_directory+'/tensorboard_logs/'+datetime.now().strftime("%Y%m%d-%H%M%S"), histogram_freq=1)

def rounded_accuracy(y_true, y_pred):
    # Round the predictions to the nearest integer
    y_pred_rounded = tf.round(y_pred)
    
    # Check if the rounded predictions match the true values
    correct_predictions = tf.cast(tf.equal(y_true, y_pred_rounded), dtype=tf.float32)
    
    # Calculate the mean accuracy
    return tf.reduce_mean(correct_predictions)


loss = keras.losses.MeanAbsoluteError()
optim = keras.optimizers.Adam(learning_rate=0.0015)
metrics = ['mae', rounded_accuracy]

model.compile(loss=loss, optimizer=optim, metrics=metrics)

batch_size = 4500
epochs = 1000

history = model.fit(X, y, epochs=epochs, batch_size=batch_size, shuffle=True, verbose=1, callbacks=[early_stopping, reduce_lr, tensorboard_callback], validation_split=0.2)

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