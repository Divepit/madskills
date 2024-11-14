import os
import numpy as np
import h5py
import tensorflow as tf
import keras
from matplotlib import pyplot as plt
from datetime import datetime
from skimage.transform import resize
from keras_tuner import RandomSearch

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
    


# Define a function to build the model with tunable hyperparameters
def build_model(hp):
    nodes = hp.Int('nodes', min_value=64, max_value=4096, step=64)
    dropout = hp.Float('dropout', min_value=0, max_value=0.5, step=0.05)
    learning_rate = hp.Float('learning_rate', min_value=0.00001, max_value=0.001, sampling='log')
    slope = hp.Float('slope', min_value=0, max_value=0.5, step=0.05)
    
    model = keras.Sequential([
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
        # keras.layers.Dropout(dropout),
        # keras.layers.Dense(nodes),
        # keras.layers.LeakyReLU(slope),
        # keras.layers.Dense(nodes),
        # keras.layers.LeakyReLU(slope),
        # keras.layers.Dense(nodes),
        # keras.layers.LeakyReLU(slope),
        keras.layers.Dense(len(y[0])),
        keras.layers.LeakyReLU(slope),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.MeanAbsoluteError(),
        metrics=['mae', rounded_accuracy]
    )
    
    return model

# Define a custom metric
def rounded_accuracy(y_true, y_pred):
    y_pred_rounded = tf.round(y_pred)
    correct_predictions = tf.cast(tf.equal(y_true, y_pred_rounded), dtype=tf.float32)
    return tf.reduce_mean(correct_predictions)

# Set up the random search tuner
tuner = RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=50,
    executions_per_trial=2,
    directory=current_directory + '/tuner_logs',
    project_name='hyperparameter_tuning'
)

# Run the hyperparameter search
batch_size = 64
tuner.search(
    X, y,
    epochs=15,
    batch_size=batch_size,
    validation_split=0.2,
    callbacks=[
        keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=0.000015)
    ]
)

# Get the best model and hyperparameters
best_model = tuner.get_best_models(1)[0]
best_hyperparameters = tuner.get_best_hyperparameters(1)[0]

# Save the best model
best_model.save(current_directory + '/best_model.keras')

print(f"Best hyperparameters: {best_hyperparameters.values}")

# Evaluate the best model
best_model.evaluate(X, y, batch_size=batch_size)