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

    # def add_activation_values(self,layer_num,G):
    #     layer_name = "Layer " + str(layer_num)
    #     if layer_name not in self.__parameters.keys():
    #         raise RuntimeError(layer_name + " isn't exist")
    #
    #     out_dim,in_dim = G.shape
    #     activation_value = {'g' + str(i) + str(j): G[i,j] for i in np.arange(start=1, stop=out_dim + 1) for j in np.arange(start=1, stop=in_dim + 1)}
    #     self.__forward[layer_name]["Activation"] = {"G":activation_value,"Dimensions":(out_dim,in_dim)}

    def add_layer_grads(self,layer_num,dA):
        layer_name = "Layer " + str(layer_num)
        if layer_name not in self.__parameters.keys():
            raise RuntimeError(layer_name + " isn't exist")

        out_dim,in_dim = dA.shape
        layer_value = {'da' + str(i) + str(j): dA[i,j] for i in np.arange(start=1, stop=out_dim + 1) for j in np.arange(start=1, stop=in_dim + 1)}
        self.__gradients[layer_name]["Layer"] = {"dA":layer_value,"Dimensions":(out_dim,in_dim)}

    def add_activation_grads(self,layer_num,dG):
        layer_name = "Layer " + str(layer_num)
        if layer_name not in self.__parameters.keys():
            raise RuntimeError(layer_name + " isn't exist")

        out_dim,in_dim = dG.shape
        activation_value = {'dg' + str(i) + str(j): dG[i,j] for i in np.arange(start=1, stop=out_dim + 1) for j in np.arange(start=1, stop=in_dim + 1)}
        self.__gradients[layer_name]["Activation"] = {"dG":activation_value,"Dimensions":(out_dim,in_dim)}

    def add_weights_grads(self,layer_num,dW):
        layer_name = "Layer " + str(layer_num)
        if layer_name not in self.__parameters.keys():
            raise RuntimeError(layer_name + " isn't exist")

        out_dim,in_dim = dW.shape
        weights_value = {'dw' + str(i) + str(j): dW[i,j] for i in np.arange(start=1, stop=out_dim + 1) for j in np.arange(start=1, stop=in_dim + 1)}
        self.__gradients[layer_name]["Parameters"] = {"dW":weights_value,"Dimensions":(out_dim,in_dim)}

    def add_bias_grads(self,layer_num,db):
        layer_name = "Layer " + str(layer_num)
        if layer_name not in self.__parameters.keys():
            raise RuntimeError(layer_name + " isn't exist")

        out_dim,in_dim = db.shape
        bias_value = {'db' + str(i) + str(j): db[i,j] for i in np.arange(start=1, stop=out_dim + 1) for j in np.arange(start=1, stop=in_dim + 1)}
        self.__gradients[layer_name]["Parameters"] = {"db":bias_value,"Dimensions":(out_dim,in_dim)}

    def get_layer_values(self,layer_num):
        layer_name = "Layer " + str(layer_num)
        if layer_name not in self.__parameters.keys():
            raise RuntimeError(layer_name + " isn't exist")

        layer_value = self.__forward[layer_name]["Layer"]["A"]
        out_dim,in_dim = self.__forward[layer_name]["Layer"]["Dimensions"]
        A = np.array([[layer_value['a' + str(row) + str(col)] for col in np.arange(start=1, stop=in_dim + 1)] for row in np.arange(start=1, stop=out_dim + 1)])

        return A

    # def get_activation_values(self,layer_num):
    #     layer_name = "Layer " + str(layer_num)
    #     if layer_name not in self.__parameters.keys():
    #         raise RuntimeError(layer_name + " isn't exist")
    #
    #     layer_value = self.__forward[layer_name]["Layer"]["G"]
    #     out_dim,in_dim = self.__forward[layer_name]["Layer"]["Dimensions"]
    #     G = np.array([[layer_value['z' + str(row) + str(col)] for col in np.arange(start=1, stop=in_dim + 1)] for row in np.arange(start=1, stop=out_dim + 1)])
    #
    #     return G

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

        layer_value = self.__gradients[layer_name]["Layer"]["dG"]
        out_dim, in_dim = self.__gradients[layer_name]["Layer"]["Dimensions"]
        dG = np.array([[layer_value['dz' + str(row) + str(col)] for col in np.arange(start=1, stop=in_dim + 1)] for row in np.arange(start=1, stop=out_dim + 1)])

        return dG

    def get_weights_grads(self, layer_num):
        layer_name = "Layer " + str(layer_num)
        if layer_name not in self.__parameters.keys():
            raise RuntimeError(layer_name + " isn't exist")

        weights_value = self.__gradients[layer_name]["Parameters"]["dW"]
        out_dim, in_dim = self.__gradients[layer_name]["Parameters"]["Dimensions"]
        dW = np.array([[weights_value['dw' + str(row) + str(col)] for col in np.arange(start=1, stop=in_dim + 1)] for row in np.arange(start=1, stop=out_dim + 1)])

        return dW

    def get_bias_grads(self, layer_num):
        layer_name = "Layer " + str(layer_num)
        if layer_name not in self.__parameters.keys():
            raise RuntimeError(layer_name + " isn't exist")

        bias_value = self.__gradients[layer_name]["Parameters"]["db"]
        out_dim, in_dim = self.__gradients[layer_name]["Parameters"]["Dimensions"]
        db = np.array([[bias_value['db' + str(row) + str(col)] for col in np.arange(start=1, stop=in_dim + 1)] for row in np.arange(start=1, stop=out_dim + 1)])

        return db

    def back_step(self,layer_num):
        dA = self.get_layer_grads(layer_num)
        dG = self.get_activation_grads(layer_num)
        dZ = np.multiply(dG,dA)

        A_prev = self.get_layer_values(layer_num - 1)
        m = A_prev.shape[1]
        dW = (1/m) * np.dot(dZ,A_prev.T)
        db = (1/m) * np.sum(dZ,axis=1,keepdims=True)

        W = self.__parameters.get_layer_weights(layer_num)
        dA_prev = np.dot(W.T,dZ)

        self.add_layer_grads(layer_num - 1,dA_prev)
        self.add_weights_grads(layer_num,dW)
        self.add_bias_grads(layer_num,db)

    def auto_propagation(self,layer_num):
        if layer_num == self.__range.start:
            return

        self.back_step(layer_num)

        self.auto_propagation(layer_num - 1)