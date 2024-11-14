import keras
from madskills.machine_learning.mlp_solver.utils import _rounded_accuracy


class MlpNetwork():
    def __init__(self,input_shape, num_output_nodes, num_nodes_per_hidden_layer, learning_rate, dropout_percentage_per_dropout_layer=0, leaky_relu_slope=0, loss=None, optimizer=None, metrics=None):
        
        self.input_shape = input_shape
        self.num_output_nodes = num_output_nodes

        self.nodes = num_nodes_per_hidden_layer
        self.dropout = dropout_percentage_per_dropout_layer
        self.learning_rate = learning_rate
        self.slope = leaky_relu_slope

        if loss is None:
            self.loss = keras.losses.MeanAbsoluteError()
        if optimizer is None:
            self.optim = keras.optimizers.Adam(learning_rate=self.learning_rate)
        if metrics is None:
            self.metrics = ['mae', 'mse', _rounded_accuracy]
        
        self.initialize_model()

    def initialize_model(self):
        self.model = keras.Sequential([
            keras.Input(shape=self.input_shape),
            keras.layers.Dense(self.nodes),
            keras.layers.LeakyReLU(negative_slope=self.slope), 
            keras.layers.Dense(self.nodes),
            keras.layers.LeakyReLU(negative_slope=self.slope),
            keras.layers.Dropout(self.dropout),
            keras.layers.Dense(self.nodes),
            keras.layers.LeakyReLU(negative_slope=self.slope),
            keras.layers.Dense(self.nodes),
            keras.layers.LeakyReLU(negative_slope=self.slope),
            keras.layers.Dropout(self.dropout),
            keras.layers.Dense(self.nodes),
            keras.layers.LeakyReLU(negative_slope=self.slope),
            keras.layers.Dense(self.nodes),
            keras.layers.LeakyReLU(negative_slope=self.slope),
            # keras.layers.Dropout(self.dropout),
            # keras.layers.Dense(self.nodes),
            # keras.layers.LeakyReLU(self.slope),
            # keras.layers.Dense(self.nodes),
            # keras.layers.LeakyReLU(self.slope),
            # keras.layers.Dense(self.nodes),
            # keras.layers.LeakyReLU(self.slope),
            keras.layers.Dense(self.num_output_nodes),
            keras.layers.ReLU(negative_slope=self.slope),
        ])

        self.model.compile(loss=self.loss, optimizer=self.optim, metrics=self.metrics)
    