class Function:
    def _init_(self, *args, **kwargs):
        # caching inputs
        self.cache = {}
        # caching gradients
        self.local_grads = {}

    # *args, **kwargs .. can pass any num/type of args to the function
    def forward(self, *args, **kwargs):
        """
            Forward Probagation - Compute function output
            return:
                Function output 
        """
        pass

    def backward(self, global_grad):
        """
            Backprobagation - Compute global gradient 
            args:
                global_grad: previous global gradient to calculate the new global gradient
            return:
                New global gradient to be backprobagated to the previous function/layer 
        """
        pass

    def calculate_local_grads(self, *args, **kwargs):
        """
            Local gradients - d_output / d_inputs
            if the function is function of many inputs, calculate the jecopian vector (gradient of each one of the inputs)
            and store it in the grads 
            return:
                local gradients of the function/layer
        """
        pass

    # Operator () overloading .. get called when calling "obj()"
    def _call_(self, *args, **kwargs):
        output = self.forward(*args, **kwargs)

        # gradient history for backprobagation
        self.local_grads = self.calculate_local_grads(*args, **kwargs)

        return output

# Layer is an extended function with weights to be trained 
# Or a function of many variables
# Could be Linear Layer or Conv2D
class Layer(Function):
    def _init_(self, *args, **kwargs):
        # call parent constructor
        super()._init_(*args, **kwargs)

        self.weights = {}
        self.weights_global_grads = {}

    def init_weights(self, *args, **kwargs):
        """
            Initialize the weights of the layer
        """
        pass


class Dense(Layer):
    
  def _init_(self,indim,outdim,*args, **kwargs):
    super()._init_()
    self.init_weights(indim,outdim)

  def init_weights(self,indim, outdim):
    """
      indim : x feature dimentions
      outdim : num of neurons in the layer
      w dims:
        (features x output_layar)
      b dims:
        (output layer x 1)
    """
   # xavier weight initialization
    self.weights['w'] = np.random.randn(indim,outdim) * np.sqrt( 2/(indim+outdim) )
    self.weights['b'] = np.zeros((outdim, 1))

  def forward(self,X):
    # print(X.shape)
    # print(self.weights['w'].shape)
    output = np.dot(self.weights['w'].T ,X) + self.weights['b']
    self.cache['x'] = X
    self.cache['output'] = output

    return output

  def backward(self,global_grad):
    """
      compute backward probagation: multiply the global gradient with the local gradient with respect to each input (W,b,X)
      args:
        global_grad: global gradient of the next layer(represent dL/dX_of_next_layer) 
          dims: (output_nuorons_of_next_layer, batch_size)
      return:
        global gradient to backprobagate to the prev layer
          dims: (output_nuorons_of_current_layer, batch_size)
    """
    batch_size = global_grad.shape[1]


    dX = np.dot(self.local_grads['x'], global_grad )

    # ========= dW dim ==========
    # dW dims = W dims .. because we have to calculate w = w - lr * dW
    # note that dW is global gradient .... but the local gradient (dY/dw) has a different dims as it is a function of the input
    # dW(x_features, output) = dw_local(x_features, batch) * global.T(batch, output)

    # ========= / batch_size .. avarage over examples =========
    # devide by batch size because avarage is calculated due to matrix multiplication of the batch raw in dw_local & batch column in global_grad.T
    # so we need to devide because the matrix mul is a sum
    dW = np.dot(np.array(self.local_grads['w']) , global_grad.T ) / batch_size
    db = np.sum(global_grad, axis = 1, keepdims = True) / batch_size

    self.weights_global_grads = {'w': dW, 'b': db}

    # =============== PRINT ====================
    # print("global=",global_grad.shape, " ..dX=",dX.shape, " .. dW_glbal=",dW.shape," .. dW_local=",np.array(self.local_grads['w']).shape)

    # return the global gradient with respect to the input(the output of the prev layer)
    return dX

  def calculate_local_grads(self, X):
    grads = {}
    grads['x'] = self.weights['w']
    grads['w'] = X
    grads['b'] = np.ones_like(self.weights['b'])
    return grads

import pandas as pd 
import pickle as cPickle
import numpy as np
import numpy
import math 
import random


# for loading CIFER-10 dataset
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo,encoding='bytes')
    return dict

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = numpy.random.permutation(len(a))
    return a[p], b[p]

class Dataset():

    def _init_(self, x, labels):
        #features
        self.x= x
        #label
        self.label= labels


    def _getitem_(self,index):
        return self.x[ : ,index ], self.label[ : ,index]

    def num_samples(self):
        pixels, samples= self.label.shape
        return samples

    def get_batch(self, batch_size, batch_iterator):
        #iterating on dataset by batch size=batch_size
        #everytime  get_batch is called it returns different number of samples
        self.x= data[batch_iterator*batch_size : batch_size*(batch_iterator+1) , 1:]
        self.label= data[batch_iterator*batch_size : batch_size*(batch_iterator+1) ,[0]]
        features=numpy.asarray(self.x)
        features=features.transpose()
        labels=numpy.asarray(self.label)
        labels=labels.transpose()
        return features, labels


    def split_data(self,ratio):
        #if ratio =0.6 we multiply it by the whole number of samples
        ratio= int(ratio* self.num_samples())
        train_features= self.x[:,:ratio]
        test_features = self.x[:,ratio:]

        train_labels = self.label[:,:ratio]
        test_labels = self.label[:,ratio :]
        train= Dataset(train_features,train_labels)
        #train.x=data[:ratio, 1:]
        #train.label=data[:ratio ,[0]]
        #train.samples= data[:ratio , :]

        test= Dataset(test_features,test_labels)
        #test.samples= data[ratio: , :]
        #test.x=data[ratio:, 1:]
        #test.label=data[ratio: ,[0]]
        return train,test



class Data_Loader():
    def _init_(self, dataset,batch_size, shuffle=0):
        features =[]
        label=[]
        no_batches = math.ceil(dataset.x.shape[1]/batch_size)


        j = dataset.x.transpose()
        l = dataset.label.transpose()

        if (shuffle == 1):
            randomize = np.arange(len(l))
            np.random.shuffle(randomize)
            l = l[randomize]
            j = j[randomize]

        s= numpy.asarray(np.array_split(j,int(no_batches)))
        b = numpy.asarray(np.array_split(l,int(no_batches)))

        for j in range(len(s)):
            features.append(s[j].transpose())
        for z in range(len(s)):
            label.append(b[z].transpose())

        self.x=features
        self.label = label


    def _getitem_(self,index):
        return self.x[index], self.label[index]


class MNIST_dataset(Dataset):

    def _init_(self,file):
        self.file=file
        images= pd.read_csv(file)
        global data
        data=images.values
        self.samples=data
        #features
        self.x= data[:, 1:].transpose()
        #label
        self.label= data[:,[0]].transpose()

        Dataset._init_(self,self.x,self.label)


class CIFER_10_dataset(Dataset):

    def _init_(self,data_dir,train_flag=1):
        
        labels= []
        feature=None
        l=[]
        if (train_flag ==1 ):
            for i in range(1):
                cifar_train_data_dict = unpickle(data_dir + "/data_batch_{}".format(i))
                if i == 1:
                    feature=cifar_train_data_dict[b'data']
                else:
                    feature=np.vstack((feature, cifar_train_data_dict[b'data']))
         
                labels.extend(cifar_train_data_dict[b'labels'])

        else: #test data
            cifar_test_data_dict = unpickle(data_dir + "/test_batch")
            feature = cifar_test_data_dict[b'data'] 
            labels= cifar_test_data_dict[b'labels']

        l.append(labels)
        feature=feature.reshape(len(feature),3,32,32)
        self.label = numpy.asarray(l)
        self.x= feature.transpose()

        Dataset._init_(self,self.x,self.label)




class CrossEntropyLoss(Loss):
    def forward(self, Y_hat, Y):
        """
            new
            yhat = (ndim, nbatch)
            y = (1, nbatch)
        """
        probs = softMax(Y_hat)
        # print(probs)
        # print()
        y = Y.T

        log_probs = -np.log([(probs[y[i], i]+1e-10) for i in range(probs.shape[1])])
        

        #  ........... Problem ...............
        # Y is inf because y hat at the begin is very big (range 8k) so e^8k = inf 
        crossentropy_loss = max(np.mean(log_probs), 0) # avrage on both axis 0 & axis 1 ()
        # crossentropy_loss = np.sum(crossentropy_loss, axis=1, keepdims=True)
        #print("Dims", probs.shape)
        
        print('Label =',Y)
        print('Pred  = ',np.argmax(probs,axis=0))

        # caching for backprop
        self.cache['probs'] = probs
        self.cache['y'] = Y
        if math.isnan(crossentropy_loss):
            sys.exit(0)
        return crossentropy_loss

    def calculate_local_grads(self, X, Y):
        probs = self.cache['probs']
        b = np.zeros((probs.shape[1],probs.shape[0]))
        b[np.arange(Y.shape[1]),Y] = 1
        b = b.T
        probs = np.subtract(probs,b) / float(Y.shape[0])
        # probs = np.sum(probs, axis=1, keepdims=True)

        #probs =  probs.mean(axis=1,keepdims=True)
        # print("back loss")
        # print(probs.shape)
        # print("What is X ?")
        # print(X.shape)
        return {'x':probs}








class GradientDecent:
    """
        Optimizer for weight update process for basic Gradient Decent algorithm
    """
    def _init_(self, parameters, learning_rate):
        # model layers
        self.parameters = parameters
        self.lr = learning_rate

    def step(self):
        """
            1 Step of weights update
        """

        for layer in self.parameters:
            # get layer's weights & d_weights  dictionaries and optimize them by various types of optimization algorithms
            for key, value in layer.weights.items():
                layer.weights[key] = self.optimize(layer.weights[key], layer.weights_global_grads[key])

    def zero_grad(self):
        for layer in self.parameters:
            for key, value in layer.weights.items():
                layer.weights_global_grads[key] = 0
        

    def optimize(self, w, dw):
        """
            Optimization Equation for different types of gradient decent 
        """
        w = w - self.lr * dw
        return w