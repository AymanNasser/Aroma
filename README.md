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

NOTE: you need to `pip install kaggle` so you need provide kaggle.json file in your environment, check this [link](https://www.kaggle.com/docs/api)

## Design
Aroma is designed based on 5 modules:

- [**nn module**](nn/): which contains the core modules of the framework such as layers, activations, losses, parameters, forward and backward modules

- [**optim module**](optim/): which contains the optimizers for updating the weights (NOTE: currently supporting just Adam and SGD)

- [**eval module**](eval/): which contains the evaluation metrices for the model

- [**vis module**](vis/): which contains the visualization module for live loss update & others

- [**utils module**](utils/): which contains the dataloader that process data for training and validation and support auto download for mnist dataset from [kaggle](https://www.kaggle.com/c/digit-recognizer), and others helper classes and functions for the framework


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

data_loader = DataLoader(batch_size=64)

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
               Softmax()], NLLLoss(), SGD(lr=0.01), live_update=False)

epoch = 16
cost = 0.
for i in range(epoch):
    for X,Y in tqdm(batches):
        y_pred = model.forward(X)
        loss = model.compute_cost(Y, y_pred)
        model.backward()
        model.step()
    print("Epoch: ", i + 1, "Loss: ", loss)

# Saving model
model.save_model()

# Evaulating model
Pred_ = model.predict(X_val)
Pred_ = np.argmax(Pred_, axis=0)
Y_val = Y_val.T.squeeze()

eval = Evaluation(Y_val, Pred_)
acc = eval.compute_accuracy()
prec = eval.compute_precision()
recall = eval.compute_recall()
f1_score = eval.compute_f1_score()
conf_mat = eval.compute_confusion_mat()
print("Accuracy: ",acc,"Precision: ",prec,"Recall: ",recall,"F1_Score: ",f1_score)   
vis = Visualization()
vis.plot_confusion_matrix(conf_mat)

```
