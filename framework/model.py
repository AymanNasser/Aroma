import numpy as np
from .losses import Loss
from .layers import Layer
from .parameters import Parameters
from .activations import Activation
from .forward import Forward
from .backpropagation import Backward


class Model:
    """
    Model module for encapsulating layers, losses & activations into a single network
    """

    def __init__(self, layers , loss : Loss, optimizer, model_name="Model_1"):
        assert isinstance(loss, Loss)
        # assert isinstance(optimizer, Optim)
        for layer in layers:
            assert isinstance(layers, Layer) or isinstance(layer, Activation)

        self.layers = layers
        layer_num = 0
        for layer in self.layers:
            if isinstance(layer, Layer):
                layer_num += 1
                layer.set_layer_num(layer_num)

            elif isinstance(layer, Activation):
                layer.set_layer_num(layer_num)
        self.loss = loss
        self.optim = optimizer
        self.model_name = model_name
        self.params = Parameters(self.model_name)
        self.__back = Backward(self.model_name,0,layer_num)
        self.__forward = Forward(self.layers,self.model_name)


    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, X):
        """
        """
        return self.forward.Propagate(X)

    def compute_cost(self,Y):
        Y_pred = self.__back.get_layer_values(len(self.layers))

        cost = self.loss.calc_loss(Y_pred,Y)
        dAL = self.loss.calc_grad(Y_pred,Y)
        self.__back.add_prediction_grads(dAL)

        return cost

    # For updating gradients
    def backward(self):
        for layer in reversed(self.layers):
            if isinstance(layer,Activation):
                G = self.__back.get_activation_values(layer.layer_num)
                dG = layer.get_grad(G)
                self.__back.add_activation_grads(layer.layer_num, dG)
                self.__back.back_step(layer.layer_num)
            else:
                dZ = self.__back.get_step_grads(layer.layer_num)
                A = self.__back.get_layer_values(layer.layer_num)
                dA_prev, dW, db = layer.backward(dZ,A)
                self.__back.add_layer_grads(layer.layer_num - 1, dA_prev)
                self.__back.add_weights_grads(layer.layer_num, dW)
                self.__back.add_bias_grads(layer.layer_num, db)

    # Setting layers grads  
    def zero_grad(self):
        pass

    # For updating params
    def step(self, learning_rate=0.01):

        for i in range(1,len(self.layers)):
            
            if isinstance(self.layers[i], Layer): # If itsn't a layer with weights & biases like linear & conv. so pass
                weights = self.params.get_layer_weights(i)
                bias = self.params.get_layer_bias(i)

                weights = weights - learning_rate*self.__back.get_weights_grads(i)
                bias = bias - learning_rate*self.__back.get_bias_grads(i)

            else:
                continue

