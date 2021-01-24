import numpy as np
from pylab import *
import seaborn as sns
from matplotlib.ticker import MaxNLocator


class Visualization:
    def __init__(self):
        self.__loss = []
        self.__epochs = 0

    def __init_fig(self):
        ion()
        self.__fig = figure()
        self.__axis = self.__fig.add_subplot(111)
                
    def plot_live_update(self,loss):
        if self.__epochs == 0:
            self.__init_fig()

        self.__epochs += 1
        self.__loss.append([loss])
        y = np.array(self.__loss)
        x = np.linspace(1, self.__epochs, num=len(self.__loss))
        self.__axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        self.__axis.plot(x,y,'.b-')
        self.__axis.set_xlabel("Batch No.")
        self.__axis.set_ylabel("Loss")
        self.__axis.set_title("Live Loss Update")
        
        draw()
        pause(0.01)

    def pause_figure(self):
        ioff()
        show()
 
    def plot_confusion_matrix(self, conf_mat):
        sns.set(font_scale=1.2) # for label size
        plt.figure(figsize = (10,7))
        sns.heatmap(conf_mat, annot=True, annot_kws={"size": 10})
        plt.show()

    def plot_sample(self, sample=(None, None), reshape_dim=(28,28,1)):
        img = sample[0]
        label = sample[1]
        img = img.reshape(reshape_dim)
        
        print(f"Sample Label: {label}")
        plt.imshow(img)
        plt.show()

