
class Forward:
    """
    Forward propagation module
    """
    def __init__(self, layers, model_name):
        # Model total layers
        self.layers = layers

        self.model_name = model_name

    def Propagate(self, X):
        """
        Propagating the input (X) of dim (batch_size, in_dim) through model layers
        """
        for layer in self.layers:
            X = layer(X)
        return X



