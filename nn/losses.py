import numpy as np

class Loss:

    def calc_loss(self, *args, **kwargs):
        pass

    def calc_grad(self, *args, **kwargs):
        pass


class MSELoss(Loss):

    def calc_loss(self, Y_pred, Y):
        total_loss = np.sum((Y_pred - Y) ** 2, axis=1, keepdims=True, dtype=np.double)
        mean_loss = total_loss / Y_pred.shape[-1]
        return mean_loss

    def calc_grad(self, Y_pred, Y):
        return 2.0 * (Y_pred - Y) / Y_pred.shape[-1]


class NLLLoss(Loss):

    def calc_loss(self, Y_pred, Y):
        """
        X is the output from fully connected layer (num_examples x num_classes)
        y is labels (num_examples x 1): one hot encode vector
        """
        m = Y.shape[-1]
        log_likelihood = np.log(Y_pred[Y, range(m)])
        loss = - np.sum(log_likelihood, dtype=np.double) / m
        return loss

    def calc_grad(self, Y_pred, Y):
        """
        X is the output from fully connected layer (num_examples x num_classes)
        y is labels (num_examples x 1): one hot encode vector
        """
        grad = np.copy(Y_pred)
        m = Y.shape[-1]
        grad[Y, range(m)] = 1.0
        grad = grad / m
        return grad


