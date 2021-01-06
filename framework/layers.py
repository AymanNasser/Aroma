import numpy as np
from .parameters import Parameters


class Layer:
    """
    Layer is the abstract model of any NN layer
    """
    # We use *args and **kwargs as an argument when we have no doubt about the number of
    # arguments we should pass in a function.
    def __init__(self, *args, **kwargs):
        # Retrieving model name
        self.model_name = kwargs['model_name']
        # Initializing params for saving & retrieving model weights & biases
        self.params = Parameters.get_model(self.model_name)
        self.layer_num = None

    def __init_weights(self, *args, **kwargs):
        """
        Initialize layer weights by a desired approach of initialization
        """
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        """
        Forward pass of the function. Calculates the output value and the
        gradient at the input as well.
        """
        raise NotImplementedError

    def backward(self, *args, **kwargs):
        """
        Backward pass. Computes the local gradient at the input value
        after forward pass.
        """
        raise NotImplementedError

    def set_layer_num(self, layer_num):
        self.layer_num = layer_num


class Linear(Layer):
    def __init__(self, in_dim, out_dim, layer_num, init_type='random'):
        super().__init__()
        self._init_weights(in_dim, out_dim)
        self.init_type = init_type

    def __init_weights(self, in_dim, out_dim):
        assert self.layer_num is None, 'Layer num is not specified'

        if self.init_type == 'random':
            self.params.initiate_random(in_dim, out_dim, self.layer_num)
        elif self.init_type == 'zero':
            self.params.initiate_zeros(in_dim, out_dim, self.layer_num)
        elif self.init_type == 'xavier':
            pass # Call init_xavier
        else:
            raise AttributeError("Non Supported Type of Init.")

    def forward(self, A_prev):
        """
        Forward pass for the Linear layer.

        Args:
            A_prev: numpy.ndarray of shape (n_batch, in_dim) containing
            the input value from the previous layer (A[l-1])

        Returns:
            None
        """
        W = self.params.get_layer_weights(self.layer_num)
        b = self.params.get_layer_bias(self.layer_num)
        Z = np.dot(W, A_prev) + b

        assert Z.shape[0] == (W.shape[0], A_prev.shape[1])

        return Z

    def backward(self, dZ):
        """
        Backward pass for linear layer

        For layer 𝑙, the linear part is: 𝑍[𝑙]=𝑊[𝑙]𝐴[𝑙−1]+𝑏[𝑙] (followed by an activation).
        The three outputs (𝑑𝑊[𝑙],𝑑𝑏[𝑙],𝑑𝐴[𝑙]) are computed using the input 𝑑𝑍[𝑙]:
            𝑑𝑊[𝑙]= 1/𝑚 * 𝑑𝑍[𝑙] * 𝐴[𝑙−1].𝑇
            𝑑𝑏[𝑙]= 1/𝑚 * sum(𝑍[𝑙](𝑖))
            𝑑𝐴[𝑙−1]= 𝑊[𝑙].𝑇 * 𝑑𝑍[𝑙]

        Args:
            dZ: Gradient of the cost with respect to the linear output (of current layer l)

        Return:
            dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
            dW -- Gradient of the cost with respect to W (current layer l), same shape as W
            db -- Gradient of the cost with respect to b (current layer l), same shape as b
        """
        A_prev = self.params.get_layer_activations(self.layer_num-1) # Must be modified by params module
        W = self.params.get_layer_weights(self.layer_num)

        m = A_prev.shape[1] # training samples

        dW = (1/m) * np.dot(dZ, A_prev.T)
        db = (1/m) * np.squeeze(np.sum(dZ, axis=1, keepdims=True))
        dA_prev = np.dot(W.T, dZ)

        assert (dA_prev.shape == A_prev.shape)
        assert (dW.shape == W.shape)

        return dA_prev, dW, db


class BatchNorm2D(Layer):
    """
    Batch normalization layer
    """
    def __init__(self, num_features, layer_num, epsilon=1e-05):
        super(self).__init__()
        self.channels = num_features
        self.layer_num = layer_num
        self.epsilon = epsilon

    def forward(self):
        pass

    def backward(self):
        pass

    def calc_grad(self):
        pass
