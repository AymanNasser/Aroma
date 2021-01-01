import numpy as np


class Activation:
    """Activation is an abstract Class of any activation function """
    def __init__(self):
        # To store the gradient of an operation
        self.grad = {}

    def __call__(self,x):
        """
        This method is called through an object instance it calculates
        the output of the function and the gradients then caches it
        """
        output = self.forward(x)
        # calculating and caching gradients
        self.grad = self.calc_grad(x)
        return output

    def forward(self, x):
        """
        Calculates the activation function output
        Args:
            x: numpy.ndarray containing
            output of a linear layer
        Returns:
            output: numpy.ndarray containing
            the activation function output
        """
        pass

    def calc_grad(self, x):
        """
        Calculates the activation function output
        Args:
             x: numpy.ndarray containing
             output of a linear layer
        Returns:
             output: numpy.ndarray containing
             the gradient of activation function operation
        """
        pass

    def get_grad(self):
        """
        Gets the gradient of an activation function
        Returns:
            self.grad{}: dictionary containing
            the gradient
        """
        return self.grad


class Sigmoid(Activation):
    def forward(self, x):
        return 1/(1 + np.exp(-x))

    def calc_grad(self, x):
        sigmOut = self.forward(x)
        sigmoid_grad = sigmOut * (1-sigmOut)
        self.grad = {"X": sigmoid_grad}
        return self.grad


class ReLU(Activation):
    def forward(self, x):
        return x * (x > 0)

    def calc_grad(self, x):
        sigmoid_grad = 1 * (x > 0)
        self.grad = {"X": sigmoid_grad}
        return self.grad


class Softmax(Activation):
    def forward(self, x):
        exp_x = np.exp(x)
        out = exp_x/np.sum(exp_x)
        return out

    def calc_grad(self, x):
        S = self.forward(x)
        S_vector = S.reshape(S.shape[0], 1)
        S_matrix = np.tile(S_vector, S.shape[0])
        S_dir = np.diag(S) - (S_matrix * np.transpose(S_matrix))
        return S_dir


x = np.array([0.1, 0.5, 0.4])

W = np.array([[0.1, 0.2, 0.3, 0.4, 0.5],
              [0.6, 0.7, 0.8, 0.9, 0.1],
              [0.11, 0.12, 0.13, 0.14, 0.15]])
Z = np.dot(np.transpose(W), x)
print(Z)
# this line instantiates sigmoid function
Y_hat = Sigmoid()
# calling the object and passing input calculates both
# the activation output and gradient of the operation
print(Y_hat(Z))
# to get the gradient dictionary
print(Y_hat.get_grad())