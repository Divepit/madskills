import os
import matplotlib.pyplot as plt
import numpy as np
import keras
from madskills.machine_learning.map_autoencoder.autoencoder_class import Autoencoder

class AutoencoderTrainer():
    def __init__(self, maps_path, save_path, save_frequency, latent_dim, val_split=0.2):
        self.save_path = save_path
        self.save_frequency = save_frequency
        self.data = None
        self.latent_dim = latent_dim
        self.val_split = val_split
        self.maps_path = maps_path
        self._init()
    
    def _init(self):
        
        assert os.path.exists(os.path.dirname(self.maps_path)), f"Path {self.maps_path} does not exist"

        self.data = list(np.load(self.maps_path, allow_pickle=True))
        print(f"Loaded {len(self.data)} maps from {self.maps_path}")
        self.data = np.array(self.data).astype('float32')
        self.data = (self.data - np.mean(self.data)) / np.max(np.abs(self.data))

        # check if not overwriting
        if os.path.exists(self.save_path):
            input(f"Model {self.save_path} already exists. Press enter to overwrite or Ctrl+C to cancel")

        split_index = int((1-self.val_split) * len(self.data))
        self.x_train, self.x_test = self.data[:split_index], self.data[split_index:]

    def train(self, epochs=10):
        self.autoencoder = Autoencoder(self.latent_dim, self.x_test.shape[1:])
        self.autoencoder.compile(optimizer='adam', loss=keras.losses.MeanSquaredError())
        self.autoencoder.fit(self.x_train, self.x_train,
                epochs=epochs,
                shuffle=True,
                validation_data=(self.x_test, self.x_test))
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        self.autoencoder.save(self.save_path)
    
    def plot_performace(self):
        encoded_imgs = self.autoencoder.encoder(self.x_test).numpy()
        decoded_imgs = self.autoencoder.decoder(encoded_imgs).numpy()
        n = 5
        plt.figure(figsize=(20, 8))
        for i in range(n):
            # display original
            ax = plt.subplot(2, n, i + 1)
            plt.imshow(self.x_test[i])
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



