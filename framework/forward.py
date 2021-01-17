from layers import Layer
from activations import Activation
from backpropagation import Backward

class Forward:
    """
    Forward propagation module
    """
    def __init__(self, layers,model_name):
        # Model total layers
        self.__layers = layers
        self.__model_name = model_name
        self.__back = Backward.get_backward_model(self.__model_name)

    def propagate(self, X):
        """
        Propagating the input (X) of dim (batch_size, in_dim) through model layers
        """
        self.__back.add_layer_values(0,X)
        for layer in self.__layers:
            X = layer.forward(X)
            if isinstance(layer,Activation):
                self.__back.add_layer_values(layer.layer_num,X)
            elif isinstance(layer,Layer):
                self.__back.add_activation_values(layer.layer_num,X)
        return X




