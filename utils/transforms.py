
class Transform:
    def __init__(self):
        pass

    def to_tensor(self, data_frame):
        return data_frame.to_numpy()

    def normalize(self, tensor):
        return (((tensor / 255.0) * 2.0) - 1.0)