import os
from madskills.machine_learning.map_autoencoder.autoencoder_trainer_class import AutoencoderTrainer

current_directory = os.path.dirname(os.path.abspath(__file__))

maps_path = current_directory + '/maps/random_maps_64.npy'
save_path = current_directory + '/models/autoencoder_64.keras'

trainer = AutoencoderTrainer(maps_path, save_path, save_frequency=10, latent_dim=16)
trainer.train(epochs=20)
trainer.plot_performace()

