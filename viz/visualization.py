import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np

class Visualization:
    def __init__(self,Image=None,cost=None,epochs=None):
        # self.Image = Image.reshape(28,28)
        # self.cost = cost
        # self.epochs = epochs
        pass


    # def make_figure(self):
    #     self.fig = plt.figure()
    #     self.fig, self.axes = plt.subplots(1, 2)

    #     # showing a sample image
    #     self.axes[1].set_title("sample image")
    #     self.axes[1].imshow(self.Image)
    #     self.axes[1].set_xticks(np.arange(0, 28, 1))
    #     self.axes[1].set_yticks(np.arange(0, 28, 1))

    #     # showing training process
    #     self.axes[0].set_title("training process")
    #     self.axes[0].set_xlabel("iterations")
    #     self.axes[0].set_ylabel("cost")
    #     self.axes[0].plot(range(1, self.epochs + 1), self.cost)
    #     self.axes[0].set_xticks(np.arange(1, self.epochs + 1, 1))
    #     self.axes[0].set_yticks(np.arange(0, max(self.cost) + 1, (max(self.cost) - min(self.cost)) / len(self.cost)))


    # def save_figure(self,figure_name):
    #     self.fig.savefig(figure_name)

    
    def plot_confusion_matrix(self, conf_mat):
        sn.set(font_scale=1.2) # for label size
        plt.figure(figsize = (10,7))
        sn.heatmap(conf_mat, annot=True, annot_kws={"size": 10})
        plt.show()
