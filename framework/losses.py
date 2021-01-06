import numpy as np


class Loss:

    def __init__(self):
        # To store the gradient of an operation
        self.grad = {}

    def __call__(self, Y_pred, Y):
        loss = self.calc_loss(Y_pred,Y)
        self.grad = self.calc_grad(Y_pred,Y)
        return loss

    def calc_loss(self, Y_pred, Y):
        pass

    def calc_grad(self, Y_pred, Y):
        pass

    def get_grad(self):
        return self.grad


class MSELoss(Loss):

    def calc_loss(self, Y_pred, Y):
        pass

    def calc_grad(self, Y_pred, Y):
        pass


class CrossEntropyLoss(Loss):

    def calc_loss(self, Y_pred, Y):
        pass

    def calc_grad(self, Y_pred, Y):
        pass
