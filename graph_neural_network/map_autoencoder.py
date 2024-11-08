import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import keras

# Parameters
amount_of_maps = 10000
current_directory = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_directory, 'maps', 'random_maps_64.npy')  # Path to save the map data
save_frequency = 100  # Save progress every 100 maps

# Ensure the directory exists
os.makedirs(os.path.dirname(file_path), exist_ok=True)

# Check if file exists to load previous maps, otherwise start fresh
if os.path.exists(file_path):
    data = list(np.load(file_path, allow_pickle=True))
    print(f"Loaded {len(data)} maps from {file_path}")
else:
    data = []
    print(f"No existing file found at {file_path}. Starting fresh.")

# Assuming `data` is a list of NumPy arrays, each with shape (N, N)
print("Number of samples:", len(data))
print("Shape of one sample:", data[0].shape)

# Convert the list to a NumPy array for efficient processing
data = np.array(data).astype('float32')
data = (data - np.mean(data)) / np.max(np.abs(data))

print(f"Data example: {data[0]}")

# Check if the conversion was successful
print("Shape after conversion to NumPy array:", data.shape)

# Split into training and testing sets (80% training, 20% testing)
split_index = int(0.8 * len(data))
x_train, x_test = data[:split_index], data[split_index:]

# Print shapes of the resulting arrays
print("Training set shape:", x_train.shape)
print("Testing set shape:", x_test.shape)


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


shape = x_test.shape[1:]
latent_dim = 8
autoencoder = Autoencoder(latent_dim, shape)

autoencoder.compile(optimizer='adam', loss=keras.losses.MeanSquaredError())

autoencoder.fit(x_train, x_train,
                epochs=10,
                shuffle=True,
                validation_data=(x_test, x_test))

encoded_imgs = autoencoder.encoder(x_test).numpy()
decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()


n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
  # display original
  ax = plt.subplot(2, n, i + 1)
  plt.imshow(x_test[i])
  plt.title("original")
  plt.gray()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)

  # display reconstruction
  ax = plt.subplot(2, n, i + 1 + n)
  plt.imshow(decoded_imgs[i])
  plt.title("reconstructed")
  plt.gray()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)
plt.show()