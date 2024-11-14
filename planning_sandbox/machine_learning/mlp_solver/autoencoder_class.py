import keras
import tensorflow as tf

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