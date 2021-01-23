

class Optimizer:
    def __init__(self, *args, **kwargs):
        pass

    def step(self, *args, **kwargs):
        raise NotImplementedError

    def init_params(self, *args, **kwargs):
        raise NotImplementedError

    def zero_grad(self, *args, **kwargs):
        raise NotImplementedError
