import numpy as np


class Loss:

    def calc_loss(self, *args, **kwargs):
        pass

    def calc_grad(self, *args, **kwargs):
        pass


class MSELoss(Loss):

    def calc_loss(self, Y_pred, Y):
        total_loss = np.sum((Y_pred - Y) ** 2, axis=1, keepdims=True)
        mean_loss = np.mean(sum)
        return mean_loss

    def calc_grad(self, Y_pred, Y):
        return 2 * (Y_pred - Y) / Y_pred.shape[0]


class CrossEntropyLoss(Loss):

    def calc_loss(self, Y_pred, Y):
        pass

    def calc_grad(self, Y_pred, Y):
        pass
