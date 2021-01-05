import numpy as np


class Parameters:
    """
        This class holds all weights for all layers for a specific model

        Features:
        - Can append for the parameters dynamically
        - Can access any parameter at any layer or neuron for fine tuning

        Initialization:
        - Initialization with zeros
        - Initialization with random
        - Initialization with xavier

        Arguments:
        - Every layer should call specific initialization and its type and pass input and output dimensions

        Constraints:
        - The output of the previous layer must equal to the input of the next layer

    """

    def __init__(self,model_name):
        self.model_name = model_name
        self.__weights = {}

    def __add_weights(self,W,b,in_dim,out_dim,layer_num):
        layer_weight = {}
        w = {'W' + str(i) + str(j): W[i - 1, j - 1] for i in np.arange(start=1, stop=out_dim + 1) for j in np.arange(start=1, stop=in_dim + 1)}
        b = {'b' + str(i): b[i - 1, 0] for i in np.arange(start=1, stop=out_dim + 1)}
        layer_weight.update(w)
        layer_weight.update(b)
        layer_name = "Layer " + str(layer_num)
        self.__weights[layer_name] = {"Weights":layer_weight,"Dimensions":(out_dim,in_dim)}

    def initiate_zeros(self,in_dim,out_dim,layer_num):
        layer_name = "Layer " + str(layer_num)
        if layer_name in self.__weights.keys():
            raise AttributeError(layer_name + " is already initialized")
        if layer_num > 1 and self.get_layer_dim(layer_num - 1)[0] != in_dim:
            raise AttributeError("Dimensions conflict in " + layer_name)
        W = np.zeros((out_dim,in_dim))
        b = np.zeros((out_dim,1))
        self.__add_weights(W,b,in_dim,out_dim,layer_num)

    def initiate_random(self,in_dim,out_dim,layer_num):
        layer_name = "Layer " + str(layer_num)
        if layer_name in self.__weights.keys():
            raise AttributeError(layer_name + " is already initialized")
        if layer_num > 1 and self.get_layer_dim(layer_num - 1)[0] != in_dim:
            raise AttributeError("Dimensions conflict in " + layer_name)
        W = np.random.randn(out_dim,in_dim)
        b = np.random.randn(out_dim,1)
        self.__add_weights(W,b,in_dim,out_dim,layer_num)

    def get_layer_weights(self,layer_num):
        layer_name = "Layer " + str(layer_num)
        if layer_name not in self.__weights.keys():
            raise RuntimeError(layer_name + " isn't exist")

        return self.__weights[layer_name]["Weights"]

    def get_layer_dim(self,layer_num):
        layer_name = "Layer " + str(layer_num)
        if layer_name not in self.__weights.keys():
            raise RuntimeError(layer_name + " isn't exist")

        return self.__weights[layer_name]["Dimensions"]


w = Parameters("test")
w.initiate_zeros(4,3,1)
w.initiate_random(3,7,2)

print(w.get_layer_weights(1))
print(w.get_layer_dim(1))