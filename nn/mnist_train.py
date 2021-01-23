from tqdm import tqdm
import os, sys

sys.path.insert(1, os.getcwd())

from utils.data import DataLoader
from nn.model import Model
from nn.activations import *
from nn.layers import *
from nn.losses import *
from optim.adam import Adam
from utils.transforms import Transform

data_loader = DataLoader(str(os.getcwd()) + '/nn',batch_size=32)

#X_train, y_train = data_loader.get_train_data(tensor_shape='4D',H=28,W=28,C=1)
X_train, y_train = data_loader.get_train_data()
trans = Transform()
X_train = trans.normalize(X_train)

batches = data_loader.get_batched_data(X_train, y_train)
x_val, y_val = data_loader.get_validation_data()

model = Model([Linear(X_train.shape[0],128, init_type='xavier'),
               ReLU(),
               Linear(128,64, init_type='xavier'),
               ReLU(),
               Linear(64,32, init_type='xavier'),
               ReLU(),
               Linear(32,10, init_type='xavier'),
               Softmax()], NLLLoss(), Adam(eps=1e-3))

#model = Model([Conv2D(1,4),Sigmoid(),MaxPool2D(),Flatten(),Linear(676,10),Softmax()],CrossEntropyLoss())
#model = Model([Conv2D(1,4),Sigmoid(),Flatten(),Linear(2704,10),Softmax()],CrossEntropyLoss())
epoch = 32

for i in range(epoch):
    for X,Y in tqdm(batches):
        y_pred = model.forward(X)
        loss = model.compute_cost(Y, y_pred)
        model.backward()
        model.step()
    print("Epoch: ", i + 1, "Loss: ", loss)

# model.forward(x_val)