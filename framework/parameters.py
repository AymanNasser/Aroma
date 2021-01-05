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

    __instances = {}

    # static method to be independent of the objects of the class, can know the parameters for different models
    @staticmethod
    def model_by_name(model_name):
        if model_name not in Parameters.__instances.keys():
            raise AttributeError("There's no parameters for model " + model_name)
        return Parameters.__instances[model_name]

    @staticmethod
    def delete_model_by_name(model_name):
        if model_name not in Parameters.__instances.keys():
            raise AttributeError("There's no parameters for model " + model_name)

        Parameters.__instances.pop(model_name)

    def get_model_name(self):
        return self.__model_name

    def __init__(self,model_name):
        if model_name in self.__class__.__instances.keys():
            raise AttributeError("The model with name " + model_name + " already exist")
        self.__model_name = model_name
        self.__parameters = {}

        # add the object of the class itself to the dictionary
        Parameters.__instances[model_name] = self

    def __add_weights(self,W,b,in_dim,out_dim,layer_num):
        layer_parameters = {}
        weights = {'W' + str(i) + str(j): W[i - 1, j - 1] for i in np.arange(start=1, stop=out_dim + 1) for j in np.arange(start=1, stop=in_dim + 1)}
        bias = {'b' + str(i): b[i - 1, 0] for i in np.arange(start=1, stop=out_dim + 1)}
        layer_parameters.update(weights)
        layer_parameters.update(bias)
        layer_name = "Layer " + str(layer_num)
        self.__parameters[layer_name] = {"Parameters":layer_parameters, "Dimensions":(out_dim, in_dim)}

    def initiate_zeros(self,in_dim,out_dim,layer_num):
        layer_name = "Layer " + str(layer_num)
        if layer_name in self.__parameters.keys():
            raise AttributeError(layer_name + " is already initialized")
        if layer_num > 1 and self.get_layer_dim(layer_num - 1)[0] != in_dim:
            raise AttributeError("Dimensions conflict in " + layer_name)
        if layer_num == 0:
            raise AttributeError("Can't initialize weights for layer zero")
        W = np.zeros((out_dim,in_dim))
        b = np.zeros((out_dim,1))
        self.__add_weights(W,b,in_dim,out_dim,layer_num)

    def initiate_random(self,in_dim,out_dim,layer_num):
        layer_name = "Layer " + str(layer_num)
        if layer_name in self.__parameters.keys():
            raise AttributeError(layer_name + " is already initialized")
        if layer_num > 1 and self.get_layer_dim(layer_num - 1)[0] != in_dim:
            raise AttributeError("Dimensions conflict in " + layer_name)
        if layer_num == 0:
            raise AttributeError("Can't initialize weights for layer zero")
        W = np.random.randn(out_dim,in_dim)
        b = np.random.randn(out_dim,1)
        self.__add_weights(W,b,in_dim,out_dim,layer_num)

    def get_layer_parameters(self,layer_num):
        layer_name = "Layer " + str(layer_num)
        if layer_name not in self.__parameters.keys():
            raise RuntimeError(layer_name + " isn't exist")

        return self.__parameters[layer_name]["Parameters"]

    def get_layer_bias(self,layer_num):
        layer_name = "Layer " + str(layer_num)
        if layer_name not in self.__parameters.keys():
            raise RuntimeError(layer_name + " isn't exist")

        parameters_dict = self.__parameters[layer_name]["Parameters"]
        out_dim, _ = self.__parameters[layer_name]["Dimensions"]
        bias = np.array([[parameters_dict['b' + str(row)]] for row in np.arange(start=1, stop=out_dim + 1)])

        return bias

    def get_layer_weights(self,layer_num):
        layer_name = "Layer " + str(layer_num)
        if layer_name not in self.__parameters.keys():
            raise RuntimeError(layer_name + " isn't exist")

        parameters_dict = self.__parameters[layer_name]["Parameters"]
        out_dim, in_dim = self.__parameters[layer_name]["Dimensions"]
        weights = np.array([[parameters_dict['W' + str(row) + str(col)] for col in np.arange(start=1, stop=in_dim + 1)] for row in np.arange(start=1, stop=out_dim + 1)])

        return weights

    def get_layer_dim(self,layer_num):
        layer_name = "Layer " + str(layer_num)
        if layer_name not in self.__parameters.keys():
            raise RuntimeError(layer_name + " isn't exist")

        return self.__parameters[layer_name]["Dimensions"]


w = Parameters("PV-RCNN")
t = Parameters("PointPillars")
w.initiate_zeros(4,3,1)
w.initiate_random(3,4,2)
t.initiate_random(5,6,1)
print(t.get_model_name(),w.get_model_name())

print(w.get_layer_parameters(2))
print(w.get_layer_weights(2))
print(w.get_layer_bias(2))
print(w.get_layer_dim(1))
print(np.array([[i for i in range(10)],[i for i in range(10)]])[1,2])