from optim.optimizer import Optimizer
from nn.parameters import Parameters
from nn.backpropagation import Backward
from nn.layers import Layer
import numpy as np


class Adam(Optimizer):
    def __init__(self, lr=0.01, betas=(0.9,0.999), eps=1e-8):
        super().__init__()
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.V = {}
        self.S = {}
        # t counts the number of steps taken of Adam 
        self.t = 0


    def init_params(self, layers, model_name):
        """
            Initializing Adams paramaters
        """
        self.model_name = model_name
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
                dW = self.__backward.get_weights_grads(i)
                db = self.__backward.get_bias_grads(i)
                
                self.V['dW' + str(i)] = beta_1 * self.V['dW' + str(i)] + (1.0 - beta_1) * dW
                self.V['db' + str(i)] = beta_1 * self.V['db' + str(i)] + (1.0 - beta_1) * db

                V_corrected['dW' + str(i)] = self.V['dW' + str(i)] / (1.0 - np.power(beta_1, self.t))
                V_corrected['db' + str(i)] = self.V['db' + str(i)] / (1.0 - np.power(beta_1, self.t))
                
                self.S['dW' + str(i)] = beta_2 * self.S['dW' + str(i)] + (1.0 - beta_2) * dW
                self.S['dW' + str(i)] = beta_2 * self.S['dW' + str(i)] + (1.0 - beta_2) * dW
                
                S_corrected['dW' + str(i)] = self.S['dW' + str(i)] / (1.0 - np.power(beta_2, self.t))
                S_corrected['db' + str(i)] = self.S['db' + str(i)] / (1.0 - np.power(beta_2, self.t))

                weights = weights - (self.lr * V_corrected['dW' + str(i)]) / np.sqrt(S_corrected['dW' + str(i)] + self.eps)
                bias = bias - (self.lr * V_corrected['db' + str(i)]) / np.sqrt(S_corrected['db' + str(i)] + self.eps)

                weights[np.isnan(weights)] = 0.0
                bias[np.isnan(bias)] = 0.0

                # Setting updated weights
                self.__params.update_layer_parameters(layer.layer_num, weights, bias)
                
            else:
                continue