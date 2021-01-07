import numpy as np

class Activation:
    """
    Activation is an abstract Class of any activation function
    """
    def __init__(self, *args, **kwargs):
        pass

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


class Sigmoid(Activation):
    def forward(self, X):
        return 1/(1 + np.exp(-X))

    def get_grad(self, X):
        sigma_out = self.forward(X)
        sigmoid_grad = sigma_out * (1-sigma_out)
        return sigmoid_grad


class ReLU(Activation):
    def forward(self, X):
        return X * (X > 0)

    def get_grad(self, X):
        relu_grad = 1 * (X > 0)
        return relu_grad


class Softmax(Activation):
    def forward(self, X):
        exp_x = np.exp(X)
        out = exp_x/np.sum(exp_x)
        return out

    def get_grad(self, X):
        S = self.forward(X)
        S_vector = S.reshape(S.shape[0], 1)
        S_matrix = np.tile(S_vector, S.shape[0])
        S_dir = np.diag(S) - (S_matrix * np.transpose(S_matrix))
        return S_dir

class LeakyRelU(Activation):
    def __init__(self, negative_slope=0.01):
        self.neg_slope = negative_slope

    def forward(self, X):
        return np.max(0, X) + self.neg_slope * np.min(0, X)

    def get_grad(self, X):
        l_relu_grad = 1 if X >= 0 else self.neg_slope
        return l_relu_grad



# x = np.array([0.1, 0.5, 0.4])
#
# W = np.array([[0.1, 0.2, 0.3, 0.4, 0.5],
#               [0.6, 0.7, 0.8, 0.9, 0.1],
#               [0.11, 0.12, 0.13, 0.14, 0.15]])
# Z = np.dot(np.transpose(W), x)
# print(Z)
# # this line instantiates sigmoid function
# Y_hat = Sigmoid()
# # calling the object and passing input calculates both
# # the activation output and gradient of the operation
# print(Y_hat(Z))
# # to get the gradient dictionary
# print(Y_hat.get_grad())