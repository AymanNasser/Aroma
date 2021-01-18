import numpy as np
from parameters import Parameters


class Layer:
    """
    Layer is the abstract model of any NN layer
    """
    # We use *args and **kwargs as an argument when we have no doubt about the number of
    # arguments we should pass in a function.
    def __init__(self, *args, **kwargs):
        # Retrieving model name
        # Initializing params for saving & retrieving model weights & biases
        self.model_name = None
        self.params = None
        self.layer_num = None

    def init_weights(self, *args, **kwargs):
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

    def set_layer_attributes(self, layer_num, model_name):
        self.layer_num = layer_num
        self.model_name = model_name
        self.params = Parameters.get_model(self.model_name)


class Linear(Layer):
    def __init__(self, in_dim, out_dim,init_type='random'):
        super().__init__()
        self.init_type = init_type
        self.in_dim = in_dim
        self.out_dim = out_dim

    def init_weights(self):

        if self.init_type == 'random':
            self.params.initiate_random(self.in_dim, self.out_dim, self.layer_num)
        elif self.init_type == 'zero':
            self.params.initiate_zeros(self.in_dim, self.out_dim, self.layer_num)
        elif self.init_type == 'xavier':
            self.params.initiate_xavier(self.in_dim, self.out_dim, self.layer_num)
        else:
            raise AttributeError("Non Supported Type of Initialization")

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
        # print(self.layer_num)
        # print(A_prev.shape, W.shape, b.shape)
        Z = np.dot(W, A_prev) + b

        # assert Z.shape == (W.shape[0], A_prev.shape[1])

        return Z

    def backward(self, dZ, A_prev):
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
        W = self.params.get_layer_weights(self.layer_num)

        m = A_prev.shape[1] # training samples
        dW = (1/m) * np.dot(dZ, A_prev.T)
        db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(W.T, dZ)

        assert dW.shape == W.shape

        return dA_prev, dW, db


class BatchNorm2D(Layer):
    """
    Batch normalization layer
    """
    def __init__(self, num_features, epsilon=1e-05):
        super(self).__init__()
        self.channels = num_features
        self.epsilon = epsilon

    def forward(self):
        pass

    def backward(self):
        pass

    def calc_grad(self):
        pass
