from nn.parameters import Parameters
from nn.backpropagation import Backward
from nn.layers import Layer
from utils.process_tensor import process_tensor
import numpy as np


class Optimizer:
    def __init__(self, lr=0.01, *args, **kwargs):
        self.lr = lr

    def step(self, *args, **kwargs):
        raise NotImplementedError

    def init_params(self, layers, model_name, *args, **kwargs):
        """
            Initializing optimizer paramaters
        """
        self.model_name = model_name
        self.params = Parameters.get_model(self.model_name)
        self.backward = Backward.get_backward_model(self.model_name)
        self.layers = layers

    def zero_grad(self, *args, **kwargs):
        raise NotImplementedError


class Adam(Optimizer):
    def __init__(self, lr=0.01, betas=(0.9,0.999), eps=1e-8):
        super().__init__(lr=lr)
        self.betas = betas
        self.eps = eps
        self.V = {}
        self.S = {}
        # t counts the number of steps taken of Adam 
        self.t = 0
    
    def init_params(self, layers, model_name):
        super().init_params(layers, model_name)
        for layer in self.layers:
            if isinstance(layer, Layer) and layer.has_weights is True:
                i = layer.layer_num
                self.V['dW' + str(i)] = np.zeros_like(self.params.get_layer_weights(i))
                self.V['db' + str(i)] = np.zeros_like(self.params.get_layer_bias(i))

                self.S['dW' + str(i)] = np.zeros_like(self.params.get_layer_weights(i))
                self.S['db' + str(i)] = np.zeros_like(self.params.get_layer_bias(i))

    def step(self):
        self.t += 1
        V_corrected = {}                         # Initializing first moment estimate, python dictionary
        S_corrected = {}                         # Initializing second moment estimate, python dictionary

        beta_1, beta_2 = self.betas
        
        for layer in self.layers:
            if isinstance(layer, Layer) and layer.has_weights is True: 
                i = layer.layer_num
                weights = self.params.get_layer_weights(i)
                bias = self.params.get_layer_bias(i)
                dW = self.backward.get_weights_grads(i)
                db = self.backward.get_bias_grads(i)
                
                self.V['dW' + str(i)] = beta_1 * self.V['dW' + str(i)] + (1.0 - beta_1) * dW
                self.V['db' + str(i)] = beta_1 * self.V['db' + str(i)] + (1.0 - beta_1) * db

                V_corrected['dW' + str(i)] = self.V['dW' + str(i)] / (1.0 - np.power(beta_1, self.t))
                V_corrected['db' + str(i)] = self.V['db' + str(i)] / (1.0 - np.power(beta_1, self.t))
                
                self.S['dW' + str(i)] = beta_2 * self.S['dW' + str(i)] + (1.0 - beta_2) * np.power(dW,2)
                self.S['db' + str(i)] = beta_2 * self.S['db' + str(i)] + (1.0 - beta_2) * np.power(db, 2)
                
                S_corrected['dW' + str(i)] = self.S['dW' + str(i)] / (1.0 - np.power(beta_2, self.t))
                S_corrected['db' + str(i)] = self.S['db' + str(i)] / (1.0 - np.power(beta_2, self.t))

                weights = weights - (self.lr * V_corrected['dW' + str(i)]) / np.sqrt(S_corrected['dW' + str(i)] + self.eps)
                bias = bias - (self.lr * V_corrected['db' + str(i)]) / np.sqrt(S_corrected['db' + str(i)] + self.eps)

                weights = process_tensor(weights)
                bias = process_tensor(bias)

                # Setting updated weights
                self.params.update_layer_parameters(layer.layer_num, weights, bias)
                
            else:
                continue


class SGD(Optimizer):
    def __init__(self, lr=0.01, momentum=0):
        super().__init__(lr=lr)
        self.beta = momentum
        if momentum != 0:
            self.V = {}

    def init_params(self, layers, model_name):
        super().init_params(layers, model_name)
        if self.beta != 0:
            for layer in self.layers:
                if isinstance(layer, Layer) and layer.has_weights is True:
                    i = layer.layer_num
                    self.V['dW' + str(i)] = np.zeros_like(self.params.get_layer_weights(i))
                    self.V['db' + str(i)] = np.zeros_like(self.params.get_layer_bias(i))


    def step(self):
        for layer in self.layers:
            if isinstance(layer, Layer) and layer.has_weights:
                i = layer.layer_num
                weights = self.params.get_layer_weights(i)
                bias = self.params.get_layer_bias(i)
                dW = self.backward.get_weights_grads(i)
                db = self.backward.get_bias_grads(i)

                if self.beta == 0:
                    weights = weights - self.lr * dW
                    bias = bias -  self.lr * db
            
                else:        
                    self.V['dW' + str(i)] = self.beta * self.V['dW' + str(i)] + (1.0 - self.beta) * dW
                    self.V['db' + str(i)] = self.beta * self.V['db' + str(i)] + (1.0 - self.beta) * db

                    weights = weights - self.lr * self.V['dW' + str(i)]
                    bias = bias - self.lr * self.V['db' + str(i)]

                # Setting updated weights
                self.params.update_layer_parameters(layer.layer_num, weights, bias)


    