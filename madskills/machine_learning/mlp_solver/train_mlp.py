# %% Train Network
import os
from madskills.machine_learning.mlp_solver.mlp_trainer_class import MlpTrainer

current_directory = os.path.dirname(os.path.abspath(__file__))
dataset_path = current_directory + '/datasets/dataset_3a_5g_random_maps.h5'
autoencoder_path = current_directory + '/models/autoencoder.keras'
tensorboard_logdir = current_directory + '/tensorboard_logs'
model_save_path = current_directory + '/models/mlp.keras'

scheduler = MlpTrainer(
    num_nodes_per_hidden_layer=1024,
    mlp_learning_rate=0.0001, 
    mlp_leaky_relu_slope = 0,mlp_dropout_percentage_per_dropout_layer=0,
    dataset_path=dataset_path,
    autoencoder_path=autoencoder_path,
    encode_maps_in_X=True,
    early_stopping=True,
    lr_schedule=True,
    tensorboard_logdir=tensorboard_logdir,
    model_save_path=model_save_path,
    overwrite_protection=False
)

scheduler.train(epochs=10, batch_size=64)