import os, sys

sys.path.insert(1, os.getcwd())

from optim.optimizer import Optimizer
from nn.parameters import Parameters
from nn.backpropagation import Backward
from nn.layers import Layer

import numpy as np
import math


class Adam(Optimizer):
    def __init__(self, lr=0.001, betas=(0.9,0.999), eps=1e-08, model_name='Model_1'):
        super().__init__()
        self.model_name
        self.lr = lr
        self.betas = betas
        self.eps = eps
        # t counts the number of steps taken of Adam 
        self.t = 0


    def init_params(self, layers):
        """
            Initializing Adams paramaters
        """
        self.__params = Parameters.get_model(self.model_name)
        self.__backward = Backward.get_backward_model(self.model_name)
        self.__layers = layers

        for layer in self.__layers:
            if isinstance(layer, Layer) and layer.has_weights is True:
                i = layer.layer_num
                self.V['dW' + str(i)] = np.zeros_like(self.__params.get_layer_weights(i))
                self.V['db' + str(i)] = np.zeros_like(self.__params.get_layer_bias(i))

                self.S['dW' + str(i)] = np.zeros_like(self.__params.get_layer_weights(i))
                self.S['db' + str(i)] = np.zeros_like(self.__params.get_layer_bias(i))


    def step(self):
        self.t += 1
        V_corrected = {}                         # Initializing first moment estimate, python dictionary
        S_corrected = {}                         # Initializing second moment estimate, python dictionary

        beta_1, beta_2 = self.betas

        for layer in self.__layers:
            if isinstance(layer, Layer) and layer.has_weights is True: 
                i = layer.layer_num

                weights = self.__params.get_layer_weights(i)
                bias = self.__params.get_layer_bias(i)
                dW = self.__back.get_weights_grads(i)
                db = self.__back.get_bias_grads(i)

                self.V['dW' + str(i)] = beta_1 * self.V['dW' + str(i)] + (1. -beta_1) * dW
                self.V['db' + str(i)] = beta_1 * self.V['db' + str(i)] + (1. -beta_1) * db

                V_corrected['dW' + str(i)] = V_corrected['dW' + str(i)] / (1. -np.power(beta_1, self.t))
                V_corrected['db' + str(i)] = V_corrected['db' + str(i)] / (1. -np.power(beta_1, self.t))
                
                self.S['dW' + str(i)] = beta_2 * self.S['dW' + str(i)] + (1. -beta_2) * dW
                self.S['dW' + str(i)] = beta_2 * self.S['dW' + str(i)] + (1. -beta_2) * dW
                
                S_corrected['dW' + str(i)] = S_corrected['dW' + str(i)] / (1. -np.power(beta_2, self.t))
                S_corrected['db' + str(i)] = S_corrected['db' + str(i)] / (1. -np.power(beta_2, self.t))

                weights = weights - self.lr * V_corrected['dW' + str(i)] / np.sqrt(S_corrected['dW' + str(i)] + self.eps)
                bias = bias - self.lr * V_corrected['db' + str(i)] / np.sqrt(S_corrected['db' + str(i)] + self.eps)
                
                # Setting updated weights
                self.__params.update_layer_parameters(layer.layer_num, weights, bias)
                
            else:
                continue

    
    