import numpy as np
from pylab import *
from matplotlib.ticker import MaxNLocator

class Visualization:
    def __init__(self, loss_type):
        self.__loss = []
        self.__epochs = 0
        self.__loss_type = loss_type
        ion()
        self.__fig = figure()
        self.__axis = self.__fig.add_subplot(111)

    def live_update(self,loss):
        self.__epochs += 1
        self.__loss.append([loss])
        y = np.array(self.__loss)
        x = np.linspace(1, self.__epochs, num=len(self.__loss))
        self.__axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        self.__axis.plot(x,y,'.b-')
        self.__axis.set_xlabel("Batch No.")
        self.__axis.set_ylabel(self.__loss_type)
        self.__axis.set_title("Live Loss Update")
        draw()
        pause(0.01)

    def pause_figure(self):
        ioff()
        show()