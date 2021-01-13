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
        self.loss = loss
        self.optim = optimizer
        self.model_name = model_name
        self.params = Parameters(self.model_name)
        self.__back = Backward(self.model_name)
        self.__forward = Forward(self.layers,self.model_name)


    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, X):
        """
        """
        return self.forward.Propagate(X)

    def compute_cost(self,Y_pred,Y):
        cost = self.loss.calc_loss(Y_pred,Y)
        dAL = self.loss.calc_grad(Y_pred,Y)
        self.__back.add_loss_grad(dAL)

        return cost

    # For updating gradients
    def backward(self):
        dAL = self.__back.get_loss_grads()
        for layer in reversed(self.layers):
            dZ = self.__back.back_step(layer.layer_num)
            if isinstance(layer,Layer):
                dA_prev,dW, db = layer.backward(dZ)
                self.__back.add_layer_grads(layer.layer_num,dA_prev)
                self.__back.add_weights_grads(layer.layer_num,dW)
                self.__back.add_bias_grads(layer.layer_num,db)
            else:
                A_prev = self.__back.get_layer_values(layer.layer_num)
                dG = layer.get_grad(A_prev)
                self.__back.add_activation_grads(layer.layer_num,dG)

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

