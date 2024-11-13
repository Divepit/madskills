import os
import numpy as np
import h5py
import tensorflow as tf
import keras
from matplotlib import pyplot as plt
from datetime import datetime
from skimage.transform import resize

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

MPP = 5
MAPSIZE = 64

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
current_directory = os.path.dirname(os.path.abspath(__file__))
dataset_path = current_directory + '/datasets/dataset_3a_5g_random_maps.h5'

autoencoder = keras.models.load_model(current_directory+'/autoencoder.keras')



def downscale_data(data):

    current_height, current_width = data.shape
    current_pixel_size = MPP * max(current_height, current_width) / max(MAPSIZE, MAPSIZE)

    scale_factor = current_pixel_size / MPP

    new_height = int(current_height / scale_factor)
    new_width = int(current_width / scale_factor)

    downscaled_data = resize(data, (new_height, new_width), order=1, mode='reflect', anti_aliasing=True)

    return downscaled_data

with h5py.File(dataset_path, 'r') as h5f:
    X = h5f['X'][:]
    y = h5f['y'][:]
    maps = h5f['map'][:]



maps = np.array([downscale_data(map) for map in maps]).astype(np.float32)
maps = (maps-np.mean(maps))/np.max(np.abs(maps))

encoded_maps = autoencoder.encoder(maps).numpy()

new_X = []
for i,encoded_map in enumerate(encoded_maps):
    new_X.append(np.concatenate((X[i], encoded_map)))

X = np.array(new_X).astype(np.float32)
    


nodes = 1024
slope = 0.1
dropout = 0.1

model = keras.Sequential(
    [
        keras.Input(shape=(X[0].shape)),
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

batch_size = 8
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