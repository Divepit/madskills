import os
import h5py
import numpy as np
import keras

from madskills.machine_learning.mlp_solver.utils import downscale_data
from madskills.machine_learning.mlp_solver.autoencoder_class import Autoencoder
from madskills.machine_learning.mlp_solver.mlp_network_class import MlpNetwork
from keras_tuner import RandomSearch
from madskills.environment.environment_class import Environment


class MlpTrainer:
    def __init__(self,
                 model_path = None,
                 num_nodes_per_hidden_layer = None,
                 mlp_learning_rate = None, 
                 mlp_leaky_relu_slope = 0,
                 mlp_dropout_percentage_per_dropout_layer=0,
                 mlp_loss=None,
                 mlp_optimizer=None,
                 mlp_metrics=None,
                 X=None,
                 y=None,
                 maps=None,
                 dataset_path=None,
                 autoencoder_path=None,
                 encode_maps_in_X=False,
                 early_stopping=False,
                 lr_schedule=False,
                 tensorboard_logdir=None,
                 model_save_path=None,
                 overwrite_protection=False,
                 ):
        assert dataset_path is None or (X is None and y is None), "Either provide X and y or a dataset_path"

        if model_path is not None:
            assert X is None and y is None and maps is None and dataset_path is None and autoencoder_path is None, "Either provide model_path or data for model creation, but never both."
            self.model = keras.models.load_model(model_path)
        else:
            self.X = X
            self.y = y
            self.maps = maps
            self.encoded_maps = None
            self.autoencoder = None
            
            self.net = None
            self.model = None
            self.num_nodes_per_hidden_layer = num_nodes_per_hidden_layer
            self.mlp_learning_rate = mlp_learning_rate
            self.mlp_dropout_percentage_per_dropout_layer = mlp_dropout_percentage_per_dropout_layer
            self.mlp_leaky_relu_slope = mlp_leaky_relu_slope
            self.mlp_loss = mlp_loss
            self.mlp_optimizer = mlp_optimizer
            self.mlp_metrics = mlp_metrics

            self.encode_maps_in_X = encode_maps_in_X
            self.overwrite_protection = overwrite_protection

            self.dataset_path = dataset_path
            if self.dataset_path is not None:
                self.import_dataset(self.dataset_path)
            

            self.autoencoder_path = autoencoder_path
            if self.autoencoder_path is not None:
                self.import_autoencoder(self.autoencoder_path)

            self._initialize_neural_net()

            self.model_save_path = model_save_path
            self.callbacks = []

            if lr_schedule:
                self.lr_schedule_callback = keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.9,
                    patience=3,
                    min_lr=0.0001,
                )
                self.callbacks.append(self.lr_schedule_callback)

            if early_stopping:
                self.early_stopping_callback = keras.callbacks.EarlyStopping(
                    patience=15,
                    restore_best_weights=True,
                )
                self.callbacks.append(self.early_stopping_callback)

            if tensorboard_logdir is not None:
                self.tensorboard_callback = keras.callbacks.TensorBoard(
                    log_dir=tensorboard_logdir,
                    histogram_freq=1,
                    profile_batch=0,
                )
                self.callbacks.append(self.tensorboard_callback)

        

    def import_dataset(self, dataset_path):
        if dataset_path is None:
            dataset_path = self.dataset_path
        with h5py.File(dataset_path, 'r') as h5f:
            self.X = h5f['X'][:]
            self.y = h5f['y'][:]
            self.maps = h5f['map'][:]

    def _initialize_neural_net(
            self,
            num_nodes_per_hidden_layer=None,
            dropout_percentage_per_dropout_layer=None,
            learning_rate=None,
            leaky_relu_slope=None,
            ):
        if num_nodes_per_hidden_layer is None:
            num_nodes_per_hidden_layer = self.num_nodes_per_hidden_layer
        if dropout_percentage_per_dropout_layer is None:
            dropout_percentage_per_dropout_layer = self.mlp_dropout_percentage_per_dropout_layer
        if learning_rate is None:
            learning_rate = self.mlp_learning_rate
        if leaky_relu_slope is None:
            leaky_relu_slope = self.mlp_leaky_relu_slope

        input_shape = self.X[0].shape
        num_output_nodes = len(self.y[0])

        self.net: MlpNetwork = MlpNetwork(
            input_shape=input_shape,
            num_output_nodes=num_output_nodes,
            num_nodes_per_hidden_layer=num_nodes_per_hidden_layer,
            dropout_percentage_per_dropout_layer=dropout_percentage_per_dropout_layer,
            learning_rate=learning_rate,
            leaky_relu_slope=leaky_relu_slope,
            loss=self.mlp_loss, 
            optimizer=self.mlp_optimizer,
            metrics=self.mlp_metrics)
        self.model = self.net.model
        return self.model
            
    def import_autoencoder(self, autoencoder_path):
        if autoencoder_path is None:
            autoencoder_path = self.autoencoder_path
        self.autoencoder = keras.models.load_model(autoencoder_path)
        if self.maps is not None:
            self._encode_maps()

    def _encode_maps(self):
        downscaled_maps = np.array([downscale_data(map, map_size=self.autoencoder.encoder.input_shape[-1]) for map in self.maps]).astype(np.float32)
        normalised_maps = (downscaled_maps-np.mean(downscaled_maps))/np.max(np.abs(downscaled_maps))
        self.encoded_maps = self.autoencoder.encoder(normalised_maps).numpy()
        if self.encode_maps_in_X:
            self._include_encoded_maps_in_X()

    def _include_encoded_maps_in_X(self):
        new_X = []
        for i,encoded_map in enumerate(self.encoded_maps):
            new_X.append(np.concatenate((self.X[i], encoded_map)))
        self.X = np.array(new_X).astype(np.float32)

    def train(self, epochs, batch_size):
        self.model.fit(self.X, self.y, epochs=epochs, batch_size=batch_size, shuffle=True, verbose=1, callbacks=self.callbacks, validation_split=0.2)
        self.save_model(model=self.model)

    def _prompt_for_path(self):
        input_path = ""
        while input_path == "":
            try:
                input_path = input("Enter path to save model. Include filename and extension (e.g. ~/models/model.keras): ")
                if input_path != "":
                    path = input_path
                else:
                    print("A path is required to save the model and move on. Use ctrl+c to cancel the saving process.")
            except KeyboardInterrupt:
                print("Not saving model.")
                break
        return path

    def save_model(self,model=None,path=None):
        if model is None:
            model = self.model
        if path is None:
            path = self.model_save_path
        if path is None:
            path = self._prompt_for_path()
        # check if no file exists at path
        if self.overwrite_protection:
            while os.path.isfile(path):
                decision = input("File already exists at path. Overwrite? (y/n): ")
                if decision == "y":
                    model.save(path)
                    break
                else:
                    path = self._prompt_for_path()
        if path is not None:
            model.save(path)

    def perform_hyperparameter_search(self, epochs, batch_size, parameter_save_path=None, model_save_path=None):
        def _hyperparameter_search(self, hp):
            nodes = hp.Int('nodes', min_value=64, max_value=4096, step=64)
            dropout = hp.Float('dropout', min_value=0, max_value=0.5, step=0.05)
            learning_rate = hp.Float('learning_rate', min_value=0.00001, max_value=0.001, sampling='log')
            slope = hp.Float('slope', min_value=0, max_value=0.5, step=0.05)

            return self._initialize_neural_net(
                num_nodes_per_hidden_layer=nodes,
                dropout_percentage_per_dropout_layer=dropout,
                learning_rate=learning_rate,
                leaky_relu_slope=slope
            )
        tuner = RandomSearch(
            _hyperparameter_search,
            objective='val_loss',
            max_trials=50,
            executions_per_trial=2,
            directory=parameter_save_path,
            project_name='hyperparameter_tuning'
        )

        callbacks_without_tensorboard = self.callbacks.copy()
        callbacks_without_tensorboard.remove(self.tensorboard_callback)

        tuner.search(
            self.X, 
            self.y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=self.callbacks.remove(self.tensorboard_callback)
        )

        self.model = tuner.get_best_models(1)[0]
        self.save_model(model=self.model, path=model_save_path)