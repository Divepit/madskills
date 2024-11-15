# %% Train Network
import os
from madskills.machine_learning.mlp_solver.mlp_trainer_class import MlpTrainer
from madskills.machine_learning.mlp_solver.utils import rounded_accuracy, penalty_mae_loss
import time

current_directory = os.path.dirname(os.path.abspath(__file__))
dataset_path = current_directory + '/datasets/dataset_3a_5g_random_maps.h5'
autoencoder_path = current_directory + '/models/autoencoder.keras'
tensorboard_logdir = current_directory + '/tensorboard_logs/'+time.strftime("%Y%m%d-%H%M%S")
model_save_path = current_directory + '/models/dataset_3a_5g_random_maps.keras'
parameter_save_path = current_directory + '/models/hyperparameters/'


mlp_trainer = MlpTrainer(
    num_nodes_per_hidden_layer=1024,
    mlp_learning_rate=0.00002, 
    mlp_leaky_relu_slope = 0,
    mlp_dropout_percentage_per_dropout_layer=0.5,
    dataset_path=dataset_path,
    autoencoder_path=autoencoder_path,
    encode_maps_in_X=True,
    early_stopping=True,
    lr_schedule=True,
    tensorboard_logdir=tensorboard_logdir,
    model_save_path=model_save_path,
    overwrite_protection=False,
    # mlp_loss=penalty_mae_loss,
    mlp_metrics=['mse','mae',rounded_accuracy,penalty_mae_loss],
)

mlp_trainer.train(epochs=1000, batch_size=500, validation_split=0.25)