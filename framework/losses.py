import numpy as np
import activations

from framework import activations


class Loss:

    def calc_loss(self, *args, **kwargs):
        pass

    def calc_grad(self, *args, **kwargs):
        pass


class MSELoss(Loss):

    def calc_loss(self, Y_pred, Y):
        total_loss = np.sum((Y_pred - Y) ** 2, axis=1, keepdims=True)
        mean_loss = np.mean(total_loss)
        return mean_loss

    def calc_grad(self, Y_pred, Y):
        return 2 * (Y_pred - Y) / Y_pred.shape[0]


class CrossEntropyLoss(Loss):

    def calc_loss(self, Y_pred, Y):
        """
        X is the output from fully connected layer (num_examples x num_classes)
        y is labels (num_examples x 1): one hot encode vector
        """
        m = Y.shape[0]
        p = activations.Softmax()
        p = p.forward(Y_pred)
        log_likelihood = -np.log(p[range(m), Y])
        loss = np.sum(log_likelihood) / m
        return loss

    def calc_grad(self, Y_pred, Y):
        """
        X is the output from fully connected layer (num_examples x num_classes)
        y is labels (num_examples x 1): one hot encode vector
        """
        m = Y.shape[0]
        grad = activations.Softmax()
        grad = grad.forward(Y_pred)
        grad[range(m), Y] -= 1
        grad = grad / m
        return grad


