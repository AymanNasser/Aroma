import numpy as np
from utils.process_tensor import process_tensor

class Activation:
    """
    Activation is an abstract Class of any activation function
    """
    def __init__(self, *args, **kwargs):
        self.layer_num = None
        self.has_weights = False

    def forward(self, *args, **kwargs):
        """
        Calculates the activation function output
        Args:
            x: numpy.ndarray containing
            output of a linear layer
        Returns:
            output: numpy.ndarray containing
            the activation function output
        """
        raise NotImplementedError


    def get_grad(self, *args, **kwargs):
        """
        Gets the gradient of an activation function
        Returns:
            self.grad{}: dictionary containing
            the gradient
        """
        raise NotImplementedError

    def set_layer_attributes(self, layer_num):
        self.layer_num = layer_num


class Sigmoid(Activation):
    def forward(self, X):
        out = 1.0/(1.0 + np.exp(-X))
        out = process_tensor(out)
        return out

    def get_grad(self, X):
        sigma_out = self.forward(X)
        sigmoid_grad = sigma_out * (1.0-sigma_out)
        sigmoid_grad = process_tensor(sigmoid_grad)
        return sigmoid_grad



class ReLU(Activation):
    def forward(self, X):
        relu = X * (X > 0)
        return relu

    def get_grad(self, X):
        relu_grad = 1.0 * (X > 0)
        return relu_grad



class Softmax(Activation):
    def forward(self, X):
        exp_x = np.exp(X)
        out = exp_x/np.sum(exp_x, axis=0, keepdims=True)
        out = process_tensor(out)
        return out

    def get_grad(self, X):
        return np.ones_like(X)


class LeakyRelU(Activation):
    def __init__(self, negative_slope=0.01):
        super().__init__()
        self.neg_slope = negative_slope

    def forward(self, X):
        out = X * (X >= 0) + (X * self.neg_slope) * (X < 0)
        return out

    def get_grad(self, X):
        grad = np.copy(X)
        grad[X >= 0] = 1.0
        grad[X < 0] = self.neg_slope
        return grad


class Tanh(Activation):
    def forward(self, X):
        tanh = (np.exp(X) - np.exp(-X)) / ((np.exp(X) + np.exp(-X)))
        tanh = process_tensor(tanh)
        return tanh
    
    def get_grad(self, X):
        tanh_out = self.forward(X)
        tanh_grad = (1.0 - np.power(tanh_out, 2))
        tanh_grad = process_tensor(tanh_grad)
        return tanh_grad