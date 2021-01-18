import numpy as np

class Evaluation:

    def __init__(self,Y,Predictions):
        self.Y = Y                                  #ground truth labels of this instance
        self.Predictions = Predictions              #calculated labels of this instance

    def get_accuracy(self):
        self.Predictions_max_index = np.argmax(self.Predictions, axis=1)
        self.accuracy = np.mean(self.Y == self.Predictions_max_index)
        return self.accuracy * 100

