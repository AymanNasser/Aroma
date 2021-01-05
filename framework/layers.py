
import numpy as np
# from params import Parameter

class Layer():
    """
    Layer is the abstract model of any NN layer
    """

    # We use *args and **kwargs as an argument when we have no doubt about the number of
    # arguments we should pass in a function.

    def __init__(self, *args, **kwargs):
        self.params = None # It should be equal to object Parameter

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def _init_weights(self, *args, **kwargs):
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def backward(self, *args, **kwargs):
        raise NotImplementedError

    def calc_grad(self, *args, **kwargs):
        raise NotImplementedError

    def get_grad(self, *args, **kwargs):
        raise NotImplementedError


class Linear(Layer):
    def __init__(self, in_dim, out_dim, layer_num, init_type='random'):
        super().__init__()
        self._init_weights(in_dim, out_dim)
        self.layer_num = layer_num
        self.init_type = init_type

    def _init_weights(self, in_dim, out_dim):
        self.params.init_weights(in_dim, out_dim, self.layer_num, self.init_type)

    def forward(self, A_prev):
        """
        Forward pass for the Linear layer.

        Args:
            A_prev: numpy.ndarray of shape (n_batch, in_dim) containing
            the input value from the previous layer (A[l-1])

        Returns:
            None
        """
        W = self.params.get_weights(self.layer_num)
        b = self.params.get_bias(self.layer_num)
        Z = np.dot(W, A_prev) + b
        assert Z.shape[0] == A_prev.shape[0]
        assert Z.shape[1] == W.shape[1] # Returning weights shape

        # Caching A, Z & b
        # Assuming cache in parameter class is a dictionary
        self.params.cache['A' + str(self.layer_num)] = A_prev
        self.params.cache['W' + str(self.layer_num)] = W
        self.params.cache['b' + str(self.layer_num)] = b
        self.params.cache['Z' + str(self.layer_num)] = Z

    def backward(self):
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
        A_prev = self.params.cache['A' + str(self.layer_num-1)]
        A = self.params.cache['A' + str(self.layer_num)]
        Y = self.params.cache['Y']
        W = self.params.cache['W' + str(self.layer_num)]
        b = self.params.cache['b' + str(self.layer_num)]

        m = A_prev.shape[1]
        dZ = (A-Y)
        dW = (1/m) * A_prev * dZ.T
        db = np.squeeze(np.sum(dZ, axis=1, keepdims=True))
        dA_prev = np.dot(W.T, dZ)

        assert (dA_prev.shape == A_prev.shape)
        assert (dW.shape == W.shape)

        # Caching grads
        self.params.cache_grads['dA' + str(self.layer_num -1)] = dA_prev
        self.params.cache_grads['dW' + str(self.layer_num )] = dW
        self.params.cache_grads['db' + str(self.layer_num)] = db

        # We should make an activation backwards
        # grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = \
        # linear_backward(sigmoid_backward())

class BatchNorm2D(Layer):
    """
    Batch normalization layer


    """
    def __init__(self, layer_num, eps=1e-05):
        super(self).__init__()
        self.layer_num = layer_num
        self.eps = eps

    def forward(self):
        pass
