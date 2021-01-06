import numpy as np
from parameters import Parameters

class Range:
    start = None
    end = None

class Backward:
    """
        This class is responsible for propagation in the backward process
    """
    __instances = {}

    # static method to be independent of the objects of the class, can know the parameters for different models
    @staticmethod
    def get_grads_model(model_name):
        if model_name not in Backward.__instances.keys():
            raise AttributeError("There's no parameters for model " + model_name)
        return Backward.__instances[model_name]

    @staticmethod
    def delete_model(model_name):
        if model_name not in Backward.__instances.keys():
            raise AttributeError("There's no parameters for model " + model_name)

        Backward.__instances.pop(model_name)

    def get_model_name(self):
        return self.__model_name


    def __init__(self,model_name,start,end):
        if model_name in self.__class__.__instances.keys():
            raise AttributeError("The model with name " + model_name + " already exist")
        self.__model_name = model_name
        self.__forward = {}
        self.__gradients = {}
        self.__parameters = Parameters.get_model(self.__model_name)
        self.__range = Range()
        self.__range.start = start
        self.__range.end = end

        # add the object of the class itself to the dictionary
        self.__class__.__instances[model_name] = self

    def add_layer_values(self,layer_num,A):
        layer_name = "Layer " + str(layer_num)
        if layer_name not in self.__parameters.keys():
            raise RuntimeError(layer_name + " isn't exist")

        out_dim,in_dim = A.shape
        layer_value = {'a' + str(i) + str(j): A[i,j] for i in np.arange(start=1, stop=out_dim + 1) for j in np.arange(start=1, stop=in_dim + 1)}
        self.__forward[layer_name]["Layer"] = {"A":layer_value,"Dimensions":(out_dim,in_dim)}

    def add_activation_values(self,layer_num,Z):
        layer_name = "Layer " + str(layer_num)
        if layer_name not in self.__parameters.keys():
            raise RuntimeError(layer_name + " isn't exist")

        out_dim,in_dim = Z.shape
        activation_value = {'z' + str(i) + str(j): Z[i,j] for i in np.arange(start=1, stop=out_dim + 1) for j in np.arange(start=1, stop=in_dim + 1)}
        self.__forward[layer_name]["Activation"] = {"Z":activation_value,"Dimensions":(out_dim,in_dim)}

    def add_layer_grads(self,layer_num,dA):
        layer_name = "Layer " + str(layer_num)
        if layer_name not in self.__parameters.keys():
            raise RuntimeError(layer_name + " isn't exist")

        out_dim,in_dim = dA.shape
        activation_value = {'da' + str(i) + str(j): dA[i,j] for i in np.arange(start=1, stop=out_dim + 1) for j in np.arange(start=1, stop=in_dim + 1)}
        self.__gradients[layer_name]["Layer"] = {"dA":activation_value,"Dimensions":(out_dim,in_dim)}

    def add_activation_grads(self,layer_num,dZ):
        layer_name = "Layer " + str(layer_num)
        if layer_name not in self.__parameters.keys():
            raise RuntimeError(layer_name + " isn't exist")

        out_dim,in_dim = dZ.shape
        layer_value = {'dz' + str(i) + str(j): dZ[i,j] for i in np.arange(start=1, stop=out_dim + 1) for j in np.arange(start=1, stop=in_dim + 1)}
        self.__gradients[layer_name]["Activation"] = {"dZ":layer_value,"Dimensions":(out_dim,in_dim)}

    def get_layer_values(self,layer_num):
        layer_name = "Layer " + str(layer_num)
        if layer_name not in self.__parameters.keys():
            raise RuntimeError(layer_name + " isn't exist")

        layer_value = self.__forward[layer_name]["Layer"]["A"]
        out_dim,in_dim = self.__forward[layer_name]["Layer"]["Dimensions"]
        A = np.array([[layer_value['a' + str(row) + str(col)] for col in np.arange(start=1, stop=in_dim + 1)] for row in np.arange(start=1, stop=out_dim + 1)])

        return A

    def get_activation_values(self,layer_num):
        layer_name = "Layer " + str(layer_num)
        if layer_name not in self.__parameters.keys():
            raise RuntimeError(layer_name + " isn't exist")

        layer_value = self.__forward[layer_name]["Layer"]["Z"]
        out_dim,in_dim = self.__forward[layer_name]["Layer"]["Dimensions"]
        Z = np.array([[layer_value['z' + str(row) + str(col)] for col in np.arange(start=1, stop=in_dim + 1)] for row in np.arange(start=1, stop=out_dim + 1)])

        return Z

    def get_layer_grads(self, layer_num):
        layer_name = "Layer " + str(layer_num)
        if layer_name not in self.__parameters.keys():
            raise RuntimeError(layer_name + " isn't exist")

        layer_value = self.__gradients[layer_name]["Layer"]["dA"]
        out_dim, in_dim = self.__gradients[layer_name]["Layer"]["Dimensions"]
        dA = np.array([[layer_value['da' + str(row) + str(col)] for col in np.arange(start=1, stop=in_dim + 1)] for row in np.arange(start=1, stop=out_dim + 1)])

        return dA

    def get_activation_grads(self, layer_num):
        layer_name = "Layer " + str(layer_num)
        if layer_name not in self.__parameters.keys():
            raise RuntimeError(layer_name + " isn't exist")

        layer_value = self.__gradients[layer_name]["Layer"]["dZ"]
        out_dim, in_dim = self.__gradients[layer_name]["Layer"]["Dimensions"]
        dZ = np.array([[layer_value['dz' + str(row) + str(col)] for col in np.arange(start=1, stop=in_dim + 1)] for row in np.arange(start=1, stop=out_dim + 1)])

        return dZ

    def back_step(self,layer_num):
        if layer_num == self.__range.start:
            return
        dA = self.get_layer_grads(layer_num)
        dZ = self.get_activation_grads(layer_num)
        dZ = np.multiply(dZ,dA)
        A_prev = self.get_layer_values(layer_num - 1)
        # To Do


        self.back_step(layer_num - 1)