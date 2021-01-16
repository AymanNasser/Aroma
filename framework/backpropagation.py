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
    def get_backward_model(model_name):
        if model_name not in Backward.__instances.keys():
            raise AttributeError("There's no parameters for model " + model_name)
        return Backward.__instances[model_name]

    @staticmethod
    def delete_backward(model_name):
        if model_name not in Backward.__instances.keys():
            raise AttributeError("There's no parameters for model " + model_name)

        Backward.__instances.pop(model_name)

    def get_backward_name(self):
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

    def __is_layer_exist(self,layer_name):
        if layer_name not in self.__forward.keys() and layer_name not in self.__gradients.keys():
            raise AttributeError(layer_name + " isn't exist")

    def __store_in_dictionary(self,layer_num,tensor,value_key,dict_key,layer_type,value_type):
        layer_name = "Layer " + str(layer_num)
        out_dim, in_dim = tensor.shape
        layer_value = {value_key + str(i) + str(j): tensor[i-1,j-1] for i in np.arange(start=1, stop=out_dim + 1) for j in np.arange(start=1, stop=in_dim + 1)}

        if layer_name not in self.__forward.keys():
            self.__forward[layer_name] = {"Layer":{},"Activation":{},"Backward":{},"Parameters":{}}
        if layer_name not in self.__gradients.keys():
            self.__gradients[layer_name] = {"Layer":{},"Activation":{},"Backward":{},"Parameters":{}}
        temp_dict = {dict_key: layer_value, "Dimensions": (out_dim, in_dim)}

        if value_type == "value":
            self.__forward[layer_name][layer_type].update(temp_dict)
        elif value_type == "gradient":
            self.__gradients[layer_name][layer_type].update(temp_dict)

    def add_layer_values(self,layer_num,A):
        self.__store_in_dictionary(layer_num, A, 'a', 'A', "Layer", "value")

    def add_activation_values(self,layer_num,G):
        self.__store_in_dictionary(layer_num, G, 'g', 'G', "Activation", "value")

    def add_prediction_values(self,AL):
        out_dim, in_dim = AL.shape
        layer_value = {'al' + str(i) + str(j): AL[i-1, j-1] for i in np.arange(start=1, stop=out_dim + 1) for j in np.arange(start=1, stop=in_dim + 1)}

        self.__forward["Prediction"] = {"AL": layer_value, "Dimensions": (out_dim, in_dim)}

    def add_prediction_grads(self,dAL):
        out_dim, in_dim = dAL.shape
        layer_value = {'dal' + str(i) + str(j): dAL[i-1, j-1] for i in np.arange(start=1, stop=out_dim + 1) for j in np.arange(start=1, stop=in_dim + 1)}

        self.__gradients["Prediction"] = {"dAL": layer_value, "Dimensions": (out_dim, in_dim)}

    def add_layer_grads(self,layer_num,dA):
        self.__store_in_dictionary(layer_num, dA, 'da', 'dA', "Layer", "gradient")

    def add_activation_grads(self,layer_num,dG):
        self.__store_in_dictionary(layer_num, dG, 'dg', 'dG', "Activation", "gradient")

    def add_step_grads(self,layer_num,dZ):
        self.__store_in_dictionary(layer_num, dZ, 'dz', 'dZ', "Backward", "gradient")

    def add_weights_grads(self,layer_num,dW):
        self.__store_in_dictionary(layer_num, dW, 'dw', 'dW', "Parameters", "gradient")

    def add_bias_grads(self,layer_num,db):
        self.__store_in_dictionary(layer_num, db, 'db', 'db', "Parameters", "gradient")

    def __get_from_dictionary(self,layer_num,value_key,dict_key,layer_type,value_type):
        layer_name = "Layer " + str(layer_num)
        self.__is_layer_exist(layer_name)

        if value_type == "value":
            layer_value = self.__forward[layer_name][layer_type][dict_key]
            out_dim, in_dim = self.__forward[layer_name][layer_type]["Dimensions"]

        elif value_type == "gradient":
            layer_value = self.__gradients[layer_name][layer_type][dict_key]
            out_dim, in_dim = self.__gradients[layer_name][layer_type]["Dimensions"]

        tensor = np.array([[layer_value[value_key + str(row) + str(col)] for col in np.arange(start=1, stop=in_dim + 1)] for row in np.arange(start=1, stop=out_dim + 1)])
        return tensor

    def get_layer_values(self,layer_num):
        return self.__get_from_dictionary(layer_num, 'a', 'A', "Layer", "value")

    def get_activation_values(self,layer_num):
        return self.__get_from_dictionary(layer_num, 'g', 'G', "Activation", "value")

    def get_prediction_values(self):
        layer_value = self.__forward["Prediction"]["AL"]
        out_dim, in_dim = self.__forward["Prediction"]["Dimensions"]

        AL = np.array([[layer_value["al" + str(row) + str(col)] for col in np.arange(start=1, stop=in_dim + 1)] for row in np.arange(start=1, stop=out_dim + 1)])
        return AL

    def get_prediction_grads(self):
        layer_value = self.__gradients["Prediction"]["dAL"]
        out_dim, in_dim = self.__gradients["Prediction"]["Dimensions"]

        dAL = np.array([[layer_value["dal" + str(row) + str(col)] for col in np.arange(start=1, stop=in_dim + 1)] for row in np.arange(start=1, stop=out_dim + 1)])
        return dAL

    def get_layer_grads(self, layer_num):
        return self.__get_from_dictionary(layer_num, 'da', 'dA', "Layer", "gradient")

    def get_activation_grads(self, layer_num):
        return self.__get_from_dictionary(layer_num, 'dg', 'dG', "Activation", "gradient")

    def get_step_grads(self, layer_num):
        return self.__get_from_dictionary(layer_num, 'dz', 'dZ', "Backward", "gradient")

    def get_weights_grads(self, layer_num):
        return self.__get_from_dictionary(layer_num, 'dw', 'dW', "Parameters", "gradient")

    def get_bias_grads(self, layer_num):
        return self.__get_from_dictionary(layer_num, 'db', 'db', "Parameters", "gradient")

    def back_step(self,layer_num):
        dA = None
        if layer_num == self.__range.end:
            dA = self.get_prediction_grads()
        else:
            dA = self.get_layer_grads(layer_num)

        dG = self.get_activation_grads(layer_num)
        dZ = np.multiply(dG,dA)

        self.add_step_grads(layer_num,dZ)

    def auto_propagation(self,layer_num):
        if layer_num == self.__range.start:
            return

        self.back_step(layer_num)

        self.auto_propagation(layer_num - 1)