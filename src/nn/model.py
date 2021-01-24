from nn.losses import Loss
from nn.layers import Layer
from nn.parameters import Parameters
from nn.activations import Activation
from nn.forward import Forward
from nn.backpropagation import Backward
from optim.optimizers import Optimizer

import os


class Model:
    """
    Model module for encapsulating layers, losses & activations into a single network
    """

    def __init__(self, layers , loss , optimizer , model_name="model"):
        assert isinstance(loss, Loss)
        assert isinstance(optimizer, Optimizer)
        for layer in layers:
            assert isinstance(layer, Layer) or isinstance(layer, Activation)

        self.__model_name = model_name
        self.__loss = loss
        self.__optim = optimizer
        self.__params = Parameters(self.__model_name)

        self.__layers = layers
        self.__layer_num = 0
        for layer in self.__layers:
            if isinstance(layer, Layer):
                self.__layer_num += 1
                layer.set_layer_attributes(self.__layer_num, self.__model_name)
                if layer.has_weights:
                    layer.init_weights()
            elif isinstance(layer, Activation):
                layer.set_layer_attributes(self.__layer_num)
            
        self.__back = Backward(self.__model_name, 0, self.__layer_num)
        self.__forward = Forward(self.__layers, self.__model_name)
        self.__optim.init_params(self.__layers, self.__model_name)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, X):
        """
        """
        return self.__forward.propagate(X)

    def compute_cost(self,Y,Y_pred):
        cost = self.__loss.calc_loss(Y_pred,Y)
        dAL = self.__loss.calc_grad(Y_pred,Y)
        self.__back.add_prediction_grads(dAL)

        return cost

    # For updating gradients
    def backward(self):
        for layer in reversed(self.__layers):
            if isinstance(layer,Activation):
                G = self.__back.get_activation_values(layer.layer_num)
                dG = layer.get_grad(G)
                self.__back.add_activation_grads(layer.layer_num, dG)
                self.__back.back_step(layer.layer_num)
            else:
                if layer.has_weights:
                    dZ = self.__back.get_step_grads(layer.layer_num,layer.has_weights)
                    A_prev = self.__back.get_layer_values(layer.layer_num - 1)
                    dA_prev, dW, db = layer.backward(dZ, A_prev)
                    self.__back.add_layer_grads(layer.layer_num - 1, dA_prev)
                    self.__back.add_weights_grads(layer.layer_num, dW)
                    self.__back.add_bias_grads(layer.layer_num, db)
                else:
                    dA = self.__back.get_step_grads(layer.layer_num, layer.has_weights)
                    A_prev = self.__back.get_layer_values(layer.layer_num - 1)
                    dA_prev = layer.backward(dA, A_prev)
                    self.__back.add_layer_grads(layer.layer_num - 1, dA_prev)

    # For updating params
    def step(self):
        self.__optim.step()

    def save_model(self, file_path=os.getcwd()):
        self.__params.save_weights(file_path)
        
    def load_model(self, file):
        if os.path.exists(file) and file.endswith('.pa'):
            self.__params.load_weights(file)
        else:
            raise AttributeError("Please specifiy correct path")

    def get_count_model_params(self):
        return self.__params.get_count_model_params()

    def predict(self, X):
        return self.__forward.predict(X)