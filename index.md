# Aroma Modules Design

Aroma is designed based on 5 modules:

[**nn module**](src/nn/): Which contains the core modules of the framework such as layers, activations, losses, parameters, forward and backward modules

[**activations**](src/nn/activations.py/):
It contains the implementations of **Sigmoid, ReLU, Softmax, LeakyRelu and tanh**

Each activation function in the module is a class which inherits from an abstract class called **Activation** which has two functions to implement, `forward()` and `get_grad()`

[**layers**](src/nn/layers.py): 
It contains the implementations of **Linear, Conv2D, MaxPool2D, AvgPool2D,Flatten, BatchNorm2D**

Each layer in the module is a class which inherits from an abstract class called **Layer** which has two functions to implement the `forward()` and `backward()` behaviour of each layer.


[**losses**](src/nn/losses.py):
It contains the implementations of **MSELoss and NLLLoss (-ve log likelihood)**

Each loss is a class which inherits from an abstract class called **Loss** which has two functions to implement `calc_loss()` for calculation of loss function and `get_grad()` to calculate the gradient of the loss

[**model**](src/nn/model.py):
It encapsulates the whole model with:
- Forward propagation `.forward()`
- Backward propagation `.backward()`
- Cost computation `.compute_cost()`
- Step for updating parameters `.step()`
- Save & load model parameters `.save()`, `.load()`
- Get no. of model’s parameters `.get_count_model_params()`
- Prediction `.predict()`

[**forward**](src/nn/forward.py):
Forwarding the input tensor through the model layers with specific caching 

[**backward**](src/nn/backpropagation.py):
Propagate backwardly through the model layers to calc. Gradients  

[**parameters**](src/nn/parameters.py): 
It caches the parameters of the module’s layers with set & get methods

