import numpy as np
from .losses import Loss
from .layers import Layer
from .parameters import Parameters
from .activations import Activation
from .forward import Forward


class Model:
    """
    Model module for encapsulating layers, losses & activations into a single network
    """

    def __init__(self, layers, loss, optimizer, model_name="Model_1"):
        assert isinstance(loss, Loss)
        # assert isinstance(optimizer, Optim)
        for layer in layers:
            assert isinstance(layers, Layer) or isinstance(layer, Activation)

        self.layers = layers
        self.loss = loss
        self.optim = optimizer
        self.model_name = model_name
        self.params = Parameters(self.model_name)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, X):
        """
        """
        forward_ = Forward(self.layers, self.model_name)
        return forward_.Propagate(X)

    def compute_cost(self):
        pass

    # For updating gradients
    def backward(self):
        pass

    def zero_grad(self):
        pass

    # For updating params
    def step(self):
        pass

