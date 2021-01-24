import numpy as np


class Transform:
    def __init__(self):
        pass

    def to_tensor(self, data_frame):
        return data_frame.to_numpy()

    def normalize(self, tensor):
        return tensor / 255.


        
