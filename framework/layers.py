import numpy as np
from .parameters import Parameters

class Layer:
    """
    Layer is the abstract model of any NN layer
    """

    # We use *args and **kwargs as an argument when we have no doubt about the number of
    # arguments we should pass in a function.
    def __init__(self, *args, **kwargs):
        # Initializing cache for intermediate results
        self.cache = {}
        # Initializing grads for backward prop.
        self.grads = {}
        # Retrieving model name
        self.model_name = kwargs['model_name']
        # Initializing params for saving & retrieving model weights & biases
        self.params = Parameters.get_model(self.model_name)
        self.layer_num = None

    def __init_weights(self, *args, **kwargs):
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

    def calc_grad(self, *args, **kwargs):
        raise NotImplementedError

    def get_grad(self, *args, **kwargs):
        raise NotImplementedError


class Linear(Layer):
    def __init__(self, in_dim, out_dim, layer_num, init_type='random'):
        super().__init__()
        self._init_weights(in_dim, out_dim)
        self.init_type = init_type

    def __init_weights(self, in_dim, out_dim):

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

        # Caching W & b
        # self.params.cache['W' + str(self.layer_num)] = W
        # self.params.cache['b' + str(self.layer_num)] = b

        # Caching A & Z for grads calc.
        self.cache['A' + str(self.layer_num)] = A_prev
        self.cache['Z' + str(self.layer_num)] = Z

        return Z

    def backward(self, A_prev, Y):
        """
        Backward pass for linear layer

        For layer 𝑙, the linear part is: 𝑍[𝑙]=𝑊[𝑙]𝐴[𝑙−1]+𝑏[𝑙] (followed by an activation).
        The three outputs (𝑑𝑊[𝑙],𝑑𝑏[𝑙],𝑑𝐴[𝑙]) are computed using the input 𝑑𝑍[𝑙]:
            𝑑𝑊[𝑙]= 1/𝑚 * 𝑑𝑍[𝑙] * 𝐴[𝑙−1].𝑇
            𝑑𝑏[𝑙]= 1/𝑚 * sum(𝑍[𝑙](𝑖))
            𝑑𝐴[𝑙−1]= 𝑊[𝑙].𝑇 * 𝑑𝑍[𝑙]

        Args: None

        Return: None
        """
        A = self.cache['A' + str(self.layer_num)]
        W = self.params.get_layer_weights(self.layer_num)
        b = self.params.get_layer_bias(self.layer_num)

        m = A_prev.shape[1] # training samples
        dZ = (A-Y)
        dW = (1/m) * A_prev * dZ.T
        db = np.squeeze(np.sum(dZ, axis=1, keepdims=True))
        dA_prev = np.dot(W.T, dZ)

        assert (dA_prev.shape == A_prev.shape)
        assert (dW.shape == W.shape)

        # Caching grads
        self.grads['dA' + str(self.layer_num -1)] = dA_prev
        self.grads['dW' + str(self.layer_num )] = dW
        self.grads['db' + str(self.layer_num)] = db

        # We should make an activation backwards
        # grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = \
        # linear_backward(sigmoid_backward())

        return dA_prev

    def calc_grad(self):
        pass


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