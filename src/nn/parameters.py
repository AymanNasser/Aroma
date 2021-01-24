import numpy as np
import pickle

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
    def get_model(model_name):
        if model_name not in Parameters.__instances.keys():
            raise AttributeError("There's no parameters for model " + model_name)
        return Parameters.__instances[model_name]

    @staticmethod
    def delete_model(model_name):
        if model_name not in Parameters.__instances.keys():
            raise AttributeError("There's no parameters for model " + model_name)

        Parameters.__instances.pop(model_name)

    def get_model_name(self):
        return self.__model_name

    def __init__(self,model_name):
        """
            @model_name: to specify which parameters belongs to which model
            @__parameters: its a dictionary to store weights and bias for each layer
                @Layer {layer_number}: the key of the dict and its value has 2 dict
                @Parameters: to store the weights {W11,W12,...} and the bias {b1,b2,...}, you can access each parameter
                             by self.__parameters[layer_name]["Parameters"]["W11"] for example
                @Dimensions: a tuple to hold the dimensions (output_dim,input_dim)

        """
        if model_name in self.__class__.__instances.keys():
            raise AttributeError("The model with name " + model_name + " already exist")
        self.__model_name = model_name
        self.__parameters_num = 0
        self.__extension = ".pa"
        self.__parameters = {}

        # add the object of the class itself to the dictionary
        self.__class__.__instances[model_name] = self

    def __check_attributes(self,layer_num):
        layer_name = "Layer " + str(layer_num)
        if layer_name in self.__parameters.keys():
            raise AttributeError(layer_name + " is already initialized")
        if layer_num == 0:
            raise AttributeError("Can't initialize weights for layer zero")

    def __is_layer_exist(self,layer_name):
        if layer_name not in self.__parameters.keys():
            raise AttributeError(layer_name + " doesn't exist")

    def __add_weights(self,Paramaeter_1,Paramaeter_2,Paramaeter_1_name,Paramaeter_2_name,layer_num):
        layer_name = "Layer " + str(layer_num)
        self.__parameters[layer_name] = {Paramaeter_1_name: np.copy(Paramaeter_1),
                                         Paramaeter_2_name: np.copy(Paramaeter_2)}

    def initiate_zeros(self,in_dim,out_dim,layer_num):
        self.__check_attributes(layer_num)
        self.__parameters_num += out_dim * (in_dim+1)
        W = np.zeros((out_dim,in_dim))
        b = np.zeros((out_dim,1))
        self.__add_weights(W,b,'W','b',layer_num)

    def initiate_random(self,in_dim,out_dim,layer_num):
        self.__check_attributes(layer_num)
        self.__parameters_num += out_dim * (in_dim + 1)
        W = np.random.randn(out_dim,in_dim)
        b = np.zeros((out_dim,1))
        self.__add_weights(W,b,'W','b',layer_num)

    def initiate_xavier(self,in_dim,out_dim,layer_num):
        self.__check_attributes(layer_num)
        self.__parameters_num += out_dim * (in_dim + 1)
        W = np.sqrt(1 / in_dim) * np.random.randn(out_dim, in_dim) 
        b = np.zeros((out_dim, 1))
        self.__add_weights(W,b,'W','b',layer_num)

    def initiate_zeros_conv2(self,f1, f2, n_C_prev, n_C,layer_num):
        self.__parameters_num += (f1*f2*n_C_prev+1) * (n_C)
        W = np.zeros((f1, f2, n_C_prev, n_C))
        b = np.zeros((1,1,1,n_C))
        self.__add_weights(W,b,'W','b',layer_num)

    def initiate_random_conv2(self, f1, f2, n_C_prev, n_C,layer_num):
        self.__parameters_num += (f1*f2*n_C_prev+1) * (n_C)
        W = np.random.randn(f1, f2, n_C_prev, n_C)
        b = np.zeros((1,1,1,n_C))
        self.__add_weights(W,b,'W','b',layer_num)

    def initiate_xavier_conv2(self,f1, f2, n_C_prev, n_C,layer_num):
        self.__parameters_num += (f1*f2*n_C_prev+1) * (n_C)
        W = np.sqrt(1 / n_C_prev) * np.random.randn(f1, f2, n_C_prev, n_C)
        b = np.zeros((1,1,1,n_C))
        self.__add_weights(W,b,'W','b',layer_num)

    def initiate_batchnorm_params(self, n_channels, layer_num):
        self.__parameters_num += 4*n_channels
        gamma = np.ones((1,1,n_channels,1))
        beta = np.zeros((1,1,n_channels,1))
        self.__add_weights(gamma,beta,'gamma','beta',layer_num)

    def get_batchnorm_params(self, layer_num):
        layer_name = "Layer " + str(layer_num)
        self.__is_layer_exist(layer_name)
        gamma = np.copy(self.__parameters[layer_name]['gamma'])
        beta = np.copy(self.__parameters[layer_name]['beta'])
        return gamma, beta

    def get_layer_bias(self,layer_num):
        layer_name = "Layer " + str(layer_num)
        self.__is_layer_exist(layer_name)
        bias = np.copy(self.__parameters[layer_name]["b"])
        return bias

    def get_layer_weights(self,layer_num):
        layer_name = "Layer " + str(layer_num)
        self.__is_layer_exist(layer_name)
        weights = np.copy(self.__parameters[layer_name]["W"])
        return weights

    def update_layer_parameters(self,layer_num,W,b):
        layer_name = "Layer " + str(layer_num)
        self.__is_layer_exist(layer_name)
        self.__add_weights(W,b,'W','b',layer_num)

    def get_layer_dim(self,layer_num):
        layer_name = "Layer " + str(layer_num)
        self.__is_layer_exist(layer_name)
        return self.__parameters[layer_name]["W"].shape

    def save_weights(self, file_path):
        file_name = self.__model_name + '_' + str(self.__parameters_num) + self.__extension
        with open(file_path + '/' + file_name, 'ab') as file:
            pickle.dump(self.__parameters, file)
        
    def load_weights(self, file):
        with open(file, 'rb') as file:
            self.__parameters = pickle.load(file)

    def get_count_model_params(self):
        return self.__parameters_num