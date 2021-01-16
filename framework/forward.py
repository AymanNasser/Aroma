from layers import Layer
from activations import Activation
from backpropagation import Backward

class Forward:
    """
    Forward propagation module
    """
    def __init__(self, layers, model_name):
        # Model total layers
        self.layers = layers
        self.model_name = model_name
        self.back = Backward.get_backward_model(self.model_name)

    def propagate(self, X):
        """
        Propagating the input (X) of dim (batch_size, in_dim) through model layers
        """
        self.back.add_layer_values(0,X)
        for layer in self.layers:
            X = layer.forward(X)
            if isinstance(layer,Activation):
                self.back.add_layer_values(layer.layer_num,X)
            elif isinstance(layer,Layer):
                self.back.add_activation_values(layer.layer_num,X)
        return X




