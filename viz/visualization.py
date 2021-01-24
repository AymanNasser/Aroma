import numpy as np
from pylab import *
import seaborn as sns
from matplotlib.ticker import MaxNLocator

class Visualization:
    def __init__(self):
        self.__loss = []
        self.__x_values = 0

    def __init_fig(self):
        ion()
        self.__fig = figure()
        self.__axis = self.__fig.add_subplot(111)
        draw()
        pause(0.01)
                
    def plot_live_update(self, xlabel, x, ylabel, y):
        if self.__x_values == 0:
            self.__init_fig()

        self.__x_values = x
        self.__loss.append([y])
        y = np.array(self.__loss)
        x = np.linspace(1, self.__x_values, num=len(self.__loss))
        self.__axis.plot(x,y,'.b-')
        self.__axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        self.__axis.set_xlabel(xlabel)
        self.__axis.set_ylabel(ylabel)
        self.__axis.set_title("Live Loss Update")
        
        draw()
        pause(0.01)

    def pause_figure(self):
        self.__loss = []
        self.__x_values = 0
        ioff()
        show()
 
    def plot_confusion_matrix(self, conf_mat):
        sns.set(font_scale=1.2) # for label size
        plt.figure(figsize = (10,7))
        sns.heatmap(conf_mat, annot=True, annot_kws={"size": 10})
        plt.show()

    def plot_sample(self, sample=None, reshape_dim=(28,28,1)):
        if isinstance(sample, tuple):
            img = sample[0]
            label = sample[1]
            print(f"Sample Label: {label}")
        else:
            img = sample
        
        img = img.reshape(reshape_dim)
        plt.imshow(img)
        plt.show()

