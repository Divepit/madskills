# %% Train Network
import os
from madskills.machine_learning.mlp_solver.mlp_trainer_class import MlpTrainer
from madskills.machine_learning.mlp_solver.utils import rounded_accuracy
import time

current_directory = os.path.dirname(os.path.abspath(__file__))
dataset_path = current_directory + '/datasets/dataset_3a_5g_random_maps.h5'
# autoencoder_path = current_directory + '/models/autoencoder.keras'
tensorboard_logdir = current_directory + '/tensorboard_logs/'+time.strftime("%Y%m%d-%H%M%S")
model_save_path = current_directory + '/models/dataset_3a_5g_random_maps.keras'
parameter_save_path = current_directory + '/models/hyperparameters/'

hyperparam_trainer = MlpTrainer(
    dataset_path=dataset_path,
    early_stopping=True,
    lr_schedule=True,
    overwrite_protection=False
)

best_hyperparameters = hyperparam_trainer.perform_hyperparameter_search(epochs=60, batch_size=500, validation_split=0.25, max_trials=20, executions_per_trial=1, parameter_save_path=parameter_save_path)

best_nodes = best_hyperparameters.get('nodes')
best_learning_rate = best_hyperparameters.get('learning_rate')
best_slope = best_hyperparameters.get('slope')
best_dropout = best_hyperparameters.get('dropout')

mlp_trainer = MlpTrainer(
    num_nodes_per_hidden_layer=best_nodes,
    mlp_learning_rate=best_learning_rate, 
    mlp_leaky_relu_slope = best_slope,
    mlp_dropout_percentage_per_dropout_layer=best_dropout,
    dataset_path=dataset_path,
    # autoencoder_path=autoencoder_path,
    encode_maps_in_X=False,
    early_stopping=True,
    lr_schedule=True,
    tensorboard_logdir=tensorboard_logdir,
    model_save_path=model_save_path,
    overwrite_protection=False,
)

mlp_trainer.train(epochs=1000, batch_size=500, validation_split=0.25)