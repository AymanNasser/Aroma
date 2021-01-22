from optimizer import Optimizer
import numpy as np
import math


class Adam(Optimizer):
    def __init__(self, parameters=None, lr=0.001, betas=(0.9,0.999), eps=1e-08):
        super().__init__()
        self.parameters = parameters
        self.lr = lr
        self.betas = betas
        self.eps = eps

    def step(self):
        pass
    