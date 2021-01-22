

class Optimizer:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError

    def step(self, *args, **kwargs):
        raise NotImplementedError

    def zero_grad(self, *args, **kwargs):
        raise NotImplementedError
