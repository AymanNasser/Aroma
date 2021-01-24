import numpy as np
from utils.process_tensor import process_tensor
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
        self.__cache = {}
        self.__range = Range()
        self.__range.start = start
        self.__range.end = end

        # add the object of the class itself to the dictionary
        self.__class__.__instances[model_name] = self

    def __is_layer_exist(self,layer_name):
        if layer_name not in self.__cache.keys():
            raise AttributeError(layer_name + " isn't exist")

    def __create_layer(self,layer_name):
        if layer_name not in self.__cache.keys():
            self.__cache[layer_name] = {}

    def __store_in_dictionary(self,layer_num,tensor,dict_key):
        layer_name = "Layer " + str(layer_num)
        self.__create_layer(layer_name)
        self.__cache[layer_name][dict_key] = process_tensor(np.copy(tensor))

    def add_layer_values(self,layer_num,A):
        self.__store_in_dictionary(layer_num,A,'A')

    def add_activation_values(self,layer_num,G):
        self.__store_in_dictionary(layer_num,G,'G')

    def add_prediction_grads(self,dA):
        self.__store_in_dictionary(self.__range.end,dA,'dA')

    def add_layer_grads(self,layer_num,dA):
        self.__store_in_dictionary(layer_num,dA,'dA')

    def add_activation_grads(self,layer_num,dG):
        self.__store_in_dictionary(layer_num,dG,'dG')

    def add_step_grads(self,layer_num,dZ):
        self.__store_in_dictionary(layer_num,dZ,'dZ')

    def add_weights_grads(self,layer_num,dW):
        self.__store_in_dictionary(layer_num,dW,'dW')

    def add_bias_grads(self,layer_num,db):
        self.__store_in_dictionary(layer_num,db,'db')

    def __get_from_dictionary(self,layer_num,dict_key):
        layer_name = "Layer " + str(layer_num)
        self.__is_layer_exist(layer_name)
        tensor = process_tensor(np.copy(self.__cache[layer_name][dict_key]))
        return tensor

    def get_layer_values(self,layer_num):
        return self.__get_from_dictionary(layer_num,"A")

    def get_activation_values(self,layer_num):
        return self.__get_from_dictionary(layer_num,"G")

    def get_prediction_grads(self):
        return self.__get_from_dictionary(self.__range.end,"dA")

    def get_layer_grads(self, layer_num):
        return self.__get_from_dictionary(layer_num,"dA")

    def get_activation_grads(self, layer_num):
        return self.__get_from_dictionary(layer_num,"dG")

    def get_step_grads(self, layer_num, has_weights):
        if has_weights:
            return self.__get_from_dictionary(layer_num,"dZ")
        else:
            return self.__get_from_dictionary(layer_num, "dA")

    def get_weights_grads(self, layer_num):
        return self.__get_from_dictionary(layer_num,"dW")

    def get_bias_grads(self, layer_num):
        return self.__get_from_dictionary(layer_num,"db")

    def back_step(self,layer_num):
        if layer_num == self.__range.end:
            dA = self.get_prediction_grads()
        else:
            dA = self.get_layer_grads(layer_num)
        dG = self.get_activation_grads(layer_num)
        dZ = np.multiply(dA,dG)
        dZ = process_tensor(dZ)
        self.add_step_grads(layer_num,dZ)