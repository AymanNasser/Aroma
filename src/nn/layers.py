import numpy as np
from nn.parameters import Parameters
from utils.process_tensor import padding

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

    def set_layer_attributes(self, *args):

        self.layer_num = args[0]
        if len(args) > 1:
            self.model_name = args[1]
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

        m = A_prev.shape[-1] # training samples
        dW = (1/m) * np.dot(dZ, A_prev.T)
        db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(W.T, dZ)

        assert dW.shape == W.shape

        return dA_prev, dW, db


class Conv2D(Layer):
    """
    Convoloution layer
    """
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0),init_type='random'):
        super().__init__()
        if isinstance(kernel_size,tuple):
            self.kernel_size = kernel_size
        elif isinstance(kernel_size,int):
            self.kernel_size = (kernel_size,kernel_size)
        else:
            raise AttributeError("Pleas specify tuple or int kernel size")

        if isinstance(stride,tuple):
            self.stride = stride
        elif isinstance(stride,int):
            self.stride = (stride,stride)
        else:
            raise AttributeError("Pleas specify tuple or int stride")

        if isinstance(padding,tuple):
            self.padding = padding
        elif isinstance(padding,int):
            self.padding = (padding,padding)
        else:
            raise AttributeError("Pleas specify tuple or int padding")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.init_type = init_type
        self.has_weights = True

    def init_weights(self):
        if self.init_type == 'random':
            self.params.initiate_random_conv2(self.kernel_size[0],self.kernel_size[1],self.in_channels,self.out_channels,self.layer_num)
        elif self.init_type == 'zero':
            self.params.initiate_zeros_conv2(self.kernel_size[0],self.kernel_size[1],self.in_channels,self.out_channels,self.layer_num)
        elif self.init_type == 'xavier':
            self.params.initiate_xavier_conv2(self.kernel_size[0],self.kernel_size[1],self.in_channels,self.out_channels,self.layer_num)
        else:
            raise AttributeError("Non Supported Type of Initialization")
    
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
        n_H = int((n_H_prev + 2 * self.padding[0] - self.kernel_size[0]) / self.stride[0]) + 1
        n_W = int((n_W_prev + 2 * self.padding[1] - self.kernel_size[1]) / self.stride[1]) + 1

        # Initialize the output volume Z with zeros. (≈1 line)
        Z = np.zeros((n_H, n_W, self.out_channels, m))
        # Create A_prev_pad by padding A_prev
        A_prev_pad = padding(A_prev, self.padding)

        for i in range(m):  # loop over the batch of training examples
            a_prev = A_prev_pad[:, :, :, i]  # Select ith training example's padded activation
            for h in range(n_H):  # loop over vertical axis of the output volume
                vert_start = h * self.stride[0]
                vert_end = h * self.stride[0] + self.kernel_size[0]
                for w in range(n_W):  # loop over horizontal axis of the output volume
                    horiz_start = w * self.stride[1]
                    horiz_end = w * self.stride[1] + self.kernel_size[1]
                    for c in range(self.out_channels):  # loop over channels (= #filters) of the output volume

                        # Use the corners to define the (3D) slice of a_prev_pad (See Hint above the cell). (≈1 line)
                        a_slice_prev = a_prev[vert_start:vert_end, horiz_start:horiz_end, :]

                        # Convolve the (3D) slice with the correct filter W and bias b, to get back one output neuron. (≈1 line)
                        Z[h, w, c, i] = self.conv_single_step(a_slice_prev, W[:, :, :, c], b[:, :, :, c])

        # Making sure your output shape is correct
        assert (Z.shape == (n_H, n_W, self.out_channels, m))
        return Z

    def backward(self, dZ, A_prev):
        """
            Implement the backward propagation for a convolution function

            Arguments:
            dZ -- gradient of the cost with respect to the output of the conv layer (Z), numpy array of shape (n_H, n_W, n_C,m)
            cache -- cache of values needed for the conv_backward(), output of conv_forward()

            Returns:
            dA_prev -- gradient of the cost with respect to the input of the conv layer (A_prev),
                       numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
            dW -- gradient of the cost with respect to the weights of the conv layer (W)
                  numpy array of shape (f, f, n_C_prev, n_C)
            db -- gradient of the cost with respect to the biases of the conv layer (b)
                  numpy array of shape (1, 1, 1, n_C)
            """

        (n_H_prev, n_W_prev, n_C_prev, m) = A_prev.shape

        # Retrieve dimensions from W's shape
        W = self.params.get_layer_weights(self.layer_num)
        (f1, f2, n_C_prev, n_C) = W.shape


        # Retrieve dimensions from dZ's shape
        (n_H, n_W, n_C, m) = dZ.shape

        # Initialize dA_prev, dW, db with the correct shapes
        dA_prev = np.zeros((n_H_prev, n_W_prev, n_C_prev,m))
        dW = np.zeros((f1, f2, n_C_prev, n_C))
        db = np.zeros((1, 1, 1, n_C))

        # Pad A_prev and dA_prev
        A_prev_pad = padding(A_prev, self.padding)
        dA_prev_pad = padding(dA_prev, self.padding)

        for i in range(m):  # loop over the training examples

            # select ith training example from A_prev_pad and dA_prev_pad
            a_prev_pad = A_prev_pad[:, :, :, i]
            da_prev_pad = dA_prev_pad[:, :, :, i]

            for h in range(n_H):  # loop over vertical axis of the output volume
                vert_start = h * self.stride[0]
                vert_end = vert_start + f1
                for w in range(n_W):  # loop over horizontal axis of the output volume
                    horiz_start = w * self.stride[1]
                    horiz_end = horiz_start + f2
                    for c in range(n_C):  # loop over the channels of the output volume

                        # Use the corners to define the slice from a_prev_pad
                        a_slice = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]

                        # Update gradients for the window and the filter's parameters using the code formulas given above
                        da_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] += W[:, :, :, c] * dZ[h, w, c,i]
                        dW[:, :, :, c] += a_slice * dZ[h, w, c, i]
                        db[:, :, :, c] += dZ[h, w, c, i]

            # Set the ith training example's dA_prev to the unpaded da_prev_pad (Hint: use X[pad:-pad, pad:-pad, :])
            if self.padding == (0, 0):
                dA_prev[:, :, :, i] = da_prev_pad
            elif self.padding[0] == 0:
                dA_prev[:, :, :, i] = da_prev_pad[:, self.padding[1]:-self.padding[1], :]
            elif self.padding[1] == 0:
                dA_prev[:, :, :, i] = da_prev_pad[self.padding[0]:-self.padding[0], :, :]
            else:
                dA_prev[:, :, :, i] = da_prev_pad[self.padding[0]:-self.padding[0], self.padding[1]:-self.padding[1], :]

        # Making sure your output shape is correct
        assert (dA_prev.shape == (n_H_prev, n_W_prev, n_C_prev, m))

        return dA_prev, dW, db


class MaxPool2D(Layer):
    """
    Max Pooling layer
    """
    def __init__(self,kernel_size=(2,2), stride=(2,2), padding=(0,0)):
        super().__init__()
        if isinstance(kernel_size,tuple):
            self.__kernel_size = kernel_size
        elif isinstance(kernel_size,int):
            self.__kernel_size = (kernel_size,kernel_size)
        else:
            raise AttributeError("Pleas specify tuple or int kernel size")

        if isinstance(stride,tuple):
            self.__stride = stride
        elif isinstance(stride,int):
            self.__stride = (stride,stride)
        else:
            raise AttributeError("Pleas specify tuple or int stride")

        if isinstance(padding,tuple):
            self.__padding = padding
        elif isinstance(padding,int):
            self.__padding = (padding,padding)
        else:
            raise AttributeError("Pleas specify tuple or int padding")

        self.has_weights = False

    def forward(self, A_prev):
        A_prev_pad = padding(A_prev, self.__padding)
        KH, KW = self.__kernel_size
        SH, SW = self.__stride
        PH, PW = self.__padding
        H, W, C, N = A_prev.shape
        A = np.zeros(((H-KH+SH+2*PH)//SH, (W-KW+SW+2*PW)//SW, C, N))
        H_out, W_out, _, _ = A.shape

        for row in range(H_out):
            h_offset = row*SH
            for col in range(W_out):
                w_offset = col*SW
                rect_field = A_prev_pad[h_offset:h_offset+KH,
                             w_offset:w_offset+KW,:,:]
                A[row, col, :, :] = np.max(rect_field, axis=(0,1))

        return A

    def backward(self, dA, A_prev):
        KH, KW = self.__kernel_size
        SH, SW = self.__stride
        H_prev, W_perv, C_prev, N_prev = A_prev.shape
        dA_prev = np.zeros((H_prev, W_perv, C_prev, N_prev))
        H, W, _, _ = dA.shape

        for row in range(H):
            h_offset = row * SH
            for col in range(W):
                w_offset = col * SW
                rect_field = A_prev[h_offset:h_offset + KH,
                             w_offset:w_offset + KW, :, :]
                mask = rect_field == np.max(rect_field, axis=(0, 1))
                dA_prev[h_offset:h_offset+KH, w_offset:w_offset+KW, :, :] = mask * dA[row, col, :, :]

        return dA_prev

class AvgPool2D(Layer):
    """
    Average Pooling layer
    """

    def __init__(self, kernel_size=(2, 2), stride=(2, 2), padding=(0, 0)):
        super().__init__()
        if isinstance(kernel_size, tuple):
            self.__kernel_size = kernel_size
        elif isinstance(kernel_size, int):
            self.__kernel_size = (kernel_size, kernel_size)
        else:
            raise AttributeError("Pleas specify tuple or int kernel size")

        if isinstance(stride, tuple):
            self.__stride = stride
        elif isinstance(stride, int):
            self.__stride = (stride, stride)
        else:
            raise AttributeError("Pleas specify tuple or int stride")

        if isinstance(padding, tuple):
            self.__padding = padding
        elif isinstance(padding, int):
            self.__padding = (padding, padding)
        else:
            raise AttributeError("Pleas specify tuple or int padding")

        self.has_weights = False

    def forward(self, A_prev):
        A_prev_pad = padding(A_prev, self.__padding)
        KH, KW = self.__kernel_size
        SH, SW = self.__stride
        PH, PW = self.__padding
        H, W, C, N = A_prev.shape
        A = np.zeros(((H - KH + SH + 2 * PH) // SH, (W - KW + SW + 2 * PW) // SW, C, N))
        H_out, W_out, _, _ = A.shape

        for row in range(H_out):
            h_offset = row*SH
            for col in range(W_out):
                w_offset = col*SW
                rect_field = A_prev_pad[h_offset:h_offset+KH,
                             w_offset:w_offset+KW,:,:]
                A[row, col, :, :] = np.average(rect_field, axis=(0, 1))

        return A

    def backward(self, dA, A_prev):
        KH, KW = self.__kernel_size
        SH, SW = self.__stride
        H_prev, W_perv, C_prev, N_prev = A_prev.shape
        dA_prev = np.zeros((H_prev, W_perv, C_prev, N_prev))
        H, W, _, _ = dA.shape

        for row in range(H):
            h_offset = row * SH
            for col in range(W):
                w_offset = col * SW
                rect_field = A_prev[h_offset:h_offset + KH,
                             w_offset:w_offset + KW, :, :]
                average = rect_field / (KH*KW)
                dA_prev[h_offset:h_offset + KH, w_offset:w_offset + KW, :, :] = average

        return dA_prev

class Flatten(Layer):
    def __init__(self):
        super().__init__()
        self.has_weights = False
    
    def forward(self, X):
        m = X.shape[-1]
        return X.reshape(-1, m)
    
    def backward(self, dZ, A):
        return dZ.reshape(A.shape)



class BatchNorm2D(Layer):
    """
    Batch normalization applies a transformation that maintains 
    the mean output close to 0 and the output standard deviation close to 1
    
    """
    def __init__(self, num_features, epsilon=1e-08):
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
        self.has_weights = True

    def init_weights(self):
        self.params.initiate_batchnorm_params(self.channels, self.layer_num)       


    def forward(self, X):
        """
            Forward pass for the 2D batchnorm layer.
            Args:
                X: numpy.ndarray of shape (height, width, n_channels, n_batch).
            Returns_
                Y: numpy.ndarray of shape (height, width, n_channels, n_batch).
                    Batch-normalized tensor of X.
        """ 
        # Prev. layer is Dense
        if len(X.shape) == 2:
            raise NotImplementedError("Didn't implement batchnorm for dense layers")
            
        
        # Prev. layer is cnn or maxpool
        elif len(X.shape) == 4:
            gamma, beta = self.params.get_batchnorm_params(self.layer_num)

            H,W,C,M = X.shape
            self.X_shape = X.shape  

            # Flattening X
            self.flat_X = X.reshape(-1,M)
            # Mean
            self.mu = np.mean(self.flat_X, axis=1, keepdims=True)
            # Variance
            self.var = np.var(self.flat_X, axis=1, keepdims=True) 
            # Normalize
            self.X_hat = (self.flat_X - self.mu) / np.sqrt(self.var + self.epsilon)
            out = gamma * self.X_hat + beta

            return out.reshape(self.X_shape)
        
        else:
            raise AttributeError("Wrong tensor dim")
        

    def backward(self, dout):
        """
            Backward pass for the 2D batchnorm layer. Calculates global gradients
            for the input and the parameters.
            Args:
                dZ_hat: numpy.ndarray of shape (height, width, n_channels, n_batch).
            Returns:
                dX: numpy.ndarray of shape (height, width, n_channels, n_batch).
                    Global gradient wrt the input X.
        """
        gamma, beta = self.params.get_batchnorm_params(self.layer_num)

        dout = dout.reshape(-1, dout.shape[-1])
        X_mu = self.flat_X - self.mu
        var_invr = 1. / np.sqrt(self.var + 1e-8)

        dBeta = np.sum(dout, axis=1)
        dGamma = np.sum(dout * self.X_hat, axis=1)

        dX_norm = dout * gamma

        dVar = np.sum(dX_norm * X_mu, axis=1) * (-0.5) * (self.var + 1e-8)**(-3 / 2)

        dMu = np.sum(dX_norm * -var_invr, axis=0) + dVar * (1 / self.X_shape[-1]) * np.sum(-2. * X_mu, axis=1)

        dX = (dX_norm * var_invr) + (dMu / self.X_shape[-1]) + (dVar * 2 / self.X_shape[-1] * X_mu)

        dX = dX.reshape(self.X_shape)
        return dX
        

    def calc_grad(self):
        pass