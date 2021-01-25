# Aroma DL-Framework
Aroma is a deep learning framework implemented in python

## Install
```
pip insall pyaroma
```

## Dependencies 
```
pip install -r requirements.txt 
```

NOTE: you need to `pip install kaggle` so you need to provide kaggle.json file in your environment, check this [link](https://www.kaggle.com/docs/api)

Aroma is designed based on 5 modules:

### NN Module
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

### Optimizer Module
[**optim module**](src/optim/): Which contains the optimizers for updating the weights ***(NOTE: currently supporting just Adam and SGD)***

[**optimizers**](src/optim/optimizers/):
It contains the implementation of two optimization algorithms **Adam and SGD**

Each algorithm is a class which inherits from an abstract class called Optimizers which has functions `step()` and `zero_grad()` to implement

### Evaluation Module
[**eval module**](src/eval/): Which contains the evaluation metrics for the model

[**evaluations**](src/eval/evaluations.py/):
It contains implementation of different evaluation metrics calculations

It has only one class called Evaluation which use two different average approaches **macro** and **weighted** 

Implemented metrics:
- Confusion matrix `.compute_confusion_mat()`
- Accuracy `.compute_accuracy()`
- F1 score: `.compute_f1_score()`
- Recall: `.compute_recall()`
- Precision: `.compute_precision()` 

[**vis module**](src/vis/visualization.py/): It contains the implementation of different visualizations

It has only one class called Visualization which has methods to visualize the update of loss function during training `.plot_live_update()`, visualize the confusion matrix `.plot_confusion_matrix()` and another to visualize a sample from a dataset `.plot_sample()`


[**utils module**](src/utils/): Which contains the data loader that process data for training, validation and testing. 


[**dataloader**](src/utils/dataloader.py/):
It has only one class that implements different functions of data preprocessing and downloading

- Download a **kaggle dataset** and loading it into a dataframe
- Split dataset into train., valid. and testing dataframes
- Reshaping data to (N x M) or (H x W x C x M)
- Partitioning data into m-batches

[**transforms**](src/utils/transforms.py/):
It has only one class called transform that implements two methods `.to_tensor()` which converts a dataframe into numpy format and another to normalize the values `.normalize()`





## Demo
```python
from nn.model import Model
from nn.activations import *
from nn.layers import *
from nn.losses import *
from optim.optimizers import Adam, SGD
from utils.dataloader import DataLoader
from eval.evaluation import Evaluation
from viz.visualization import Visualization
from utils.transforms import Transform
from tqdm import tqdm

INPUT_FEATURE = 784

data_loader = DataLoader(batch_size=64, dataset_path="../")

# Training
X_train, y_train = data_loader.get_train_data()
trans = Transform()
X_train = trans.normalize(X_train)
batches = data_loader.get_batched_data(X_train, y_train)

# Validation
X_val, Y_val = data_loader.get_validation_data()
X_val = trans.normalize(X_val)


model = Model([Linear(INPUT_FEATURE,128, init_type='xavier'),
               ReLU(),
               Linear(128,64, init_type='xavier'),
               ReLU(),
               Linear(64,32, init_type='xavier'),
               ReLU(),
               Linear(32,16, init_type='xavier'),
               ReLU(),
               Linear(16,10, init_type='xavier'),
               Softmax()], loss=NLLLoss(), optimizer=Adam(lr=0.001))

epoch = 16

vis = Visualization()

for i in range(epoch):
    for X,Y in tqdm(batches):
        y_pred = model.forward(X)
        loss = model.compute_cost(Y, y_pred)
        model.backward()
        model.step()
    vis.plot_live_update(xlabel="Epoch No.", x=i + 1, ylabel="Loss", y=loss)

vis.pause_figure()

model.save_model()

# Evaluating model
Pred_ = model.predict(X_val)
Pred_ = np.argmax(Pred_, axis=0)
Y_val = Y_val.T.squeeze()

eval = Evaluation(Y_val, Pred_, average='weighted')
acc = eval.compute_accuracy()
prec = eval.compute_precision()
recall = eval.compute_recall()
f1_score = eval.compute_f1_score()
conf_mat = eval.compute_confusion_mat()
print("Accuracy: ",acc,"Precision: ",prec,"Recall: ",recall,"F1_Score: ",f1_score)   

vis.plot_confusion_matrix(conf_mat)

```
