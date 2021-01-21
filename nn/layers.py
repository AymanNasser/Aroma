import numpy as np
from nn.parameters import Parameters


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
        self.has_weights = None

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
        self.params = Parameters.get_model(self.model_name) # Need to be set if only the layer has parameters


class Linear(Layer):
    def __init__(self, in_dim, out_dim,init_type='random'):
        super().__init__()
        self.init_type = init_type
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.has_weights = True

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
        Z = np.dot(W, A_prev) + b

        assert Z.shape == (W.shape[0], A_prev.shape[1])

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


class Conv2D(Layer):
    """
    Convoloution layer
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,init_type='random'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.kernel_size = kernel_size
        self.padding = padding
        self.init_type = init_type
        self.has_weights = True

    def init_weights(self):
        if self.init_type == 'random':
            self.params.initiate_random_conv2(self.kernel_size,self.kernel_size,self.in_channels,self.out_channels,self.layer_num)
        elif self.init_type == 'zero':
            self.params.initiate_zeros_conv2(self.kernel_size,self.kernel_size,self.in_channels,self.out_channels,self.layer_num)
        elif self.init_type == 'xavier':
            self.params.initiate_xavier_conv2(self.kernel_size,self.kernel_size,self.in_channels,self.out_channels,self.layer_num)
        else:
            raise AttributeError("Non Supported Type of Initialization")

    def init_weights(self):
        pass
    
    def conv_single_step(self, a_slice_prev, W, b):
        """
        Apply one filter defined by parameters W on a single slice (a_slice_prev) of the output activation
        of the previous layer.

        Arguments:
        a_slice_prev -- slice of input data of shape (f, f, n_C_prev)
        W -- Weight parameters contained in a window - matrix of shape (f, f, n_C_prev)
        b -- Bias parameters contained in a window - matrix of shape (1, 1, 1)

        Returns:
        Z -- a scalar value, result of convolving the sliding window (W, b) on a slice x of the input data
        """

        # Element-wise product between a_slice and W. Do not add the bias yet.
        s = np.multiply(a_slice_prev, W)
        # Sum over all entries of the volume s.
        Z = np.sum(s)
        # Add bias b to Z. Cast b to a float() so that Z results in a scalar value.
        Z = Z + b.astype(float)

        return Z

    def forward(self, A_prev):
        """
            Implements the forward propagation for a convolution function

            Arguments:
            A_prev -- output activations of the previous layer, numpy array of shape (n_H_prev, n_W_prev, n_C_prev,m)
            W -- Weights, numpy array of shape (f, f, n_C_prev, n_C)
            b -- Biases, numpy array of shape (1, 1, 1, n_C)

            Returns:
            Z -- conv output, numpy array of shape (n_H, n_W, n_C,m)
        """
        W = self.params.get_layer_weights(self.layer_num)
        b = self.params.get_layer_bias(self.layer_num)
        (n_H_prev, n_W_prev, n_C_prev, m) = A_prev.shape

        # calculating output dimensions
        n_H = int((n_H_prev + 2 * self.padding - self.kernel_size) / self.stride) + 1
        n_W = int((n_W_prev + 2 * self.padding - self.kernel_size) / self.stride) + 1

        # Initialize the output volume Z with zeros. (≈1 line)
        Z = np.zeros([n_H, n_W, self.out_channels, m])
        # Create A_prev_pad by padding A_prev
        #A_prev_pad = zero_pad(A_prev, pad)

        for i in range(m):  # loop over the batch of training examples
            a_prev = A_prev[:, :, :, i]  # Select ith training example's padded activation
            for h in range (n_H):  # loop over vertical axis of the output volume
                for w in range(n_W):  # loop over horizontal axis of the output volume
                    for c in range(self.out_channels):  # loop over channels (= #filters) of the output volume

                        # Find the corners of the current "slice" (≈4 lines)
                        vert_start = h * self.stride
                        vert_end = h * self.stride + self.kernel_size
                        horiz_start = w * self.stride
                        horiz_end = w * self.stride + self.kernel_size

                        # Use the corners to define the (3D) slice of a_prev_pad (See Hint above the cell). (≈1 line)
                        a_slice_prev = a_prev[vert_start:vert_end,horiz_start:horiz_end,:]

                        # Convolve the (3D) slice with the correct filter W and bias b, to get back one output neuron. (≈1 line)
                        Z[h, w, c, i] = self.conv_single_step(a_slice_prev, W[:, :, :, c], b[:, :, :, c])


        # Making sure your output shape is correct
        assert (Z.shape == (n_H, n_W, self.out_channels, m))
        return Z

    def backward(self, *args, **kwargs):
        pass


class BatchNorm2D(Layer):
    """
    Batch normalization applies a transformation that maintains 
    the mean output close to 0 and the output standard deviation close to 1
    """
    def __init__(self, num_features, epsilon=1e-05, axis=2, affine=True):
        super().__init__()
        """
        Parameters:
            num_features: C from an expected input of size (W,H,C,M)
            affine – a boolean value that when set to True, this module has learnable affine parameters. Default: True
            epsilon – a value added to the denominator for numerical stability. Default: 1e-5
            axis = an int, the axis that should be normalized (typically the features axis)
        """
        self.channels = num_features
        self.epsilon = epsilon
        self.axis = axis
        self.affine = affine
        self.has_weights = True

    def init_weights(self):
        pass

    def forward(self, X):
        pass

    def backward(self):
        pass

    def calc_grad(self):
        pass
