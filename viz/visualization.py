import matplotlib.pyplot as plt
import numpy as np

class Visualization:
    def __init__(self,Image,cost,epochs):
        self.Image = Image.reshape(28,28)
        self.cost = cost
        self.epochs = epochs


    def make_figure(self):
        self.fig = plt.figure()
        self.fig, self.axes = plt.subplots(1, 2)

        # showing a sample image
        self.axes[1].set_title("sample image")
        self.axes[1].imshow(self.Image)
        self.axes[1].set_xticks(np.arange(0, 28, 1))
        self.axes[1].set_yticks(np.arange(0, 28, 1))

        # showing training process
        self.axes[0].set_title("training process")
        self.axes[0].set_xlabel("iterations")
        self.axes[0].set_ylabel("cost")
        self.axes[0].plot(range(1, self.epochs + 1), self.cost)
        self.axes[0].set_xticks(np.arange(1, self.epochs + 1, 1))
        self.axes[0].set_yticks(np.arange(0, max(self.cost) + 1, (max(self.cost) - min(self.cost)) / len(self.cost)))


    def save_figure(self,figure_name):
        self.fig.savefig(figure_name)



