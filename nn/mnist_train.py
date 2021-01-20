from tqdm import tqdm

from utils.data import DataLoader
from nn.model import Model
from nn.activations import *
from nn.layers import *
from nn.losses import *


data_loader = DataLoader(batch_size=64)
X_train, y_train = data_loader.get_train_data()

y_train = y_train.reshape(y_train.shape[0],1)
batches = data_loader.get_batched_data(X_train, y_train)
y_train = y_train.T

model = Model([Linear(X_train.shape[0],10),Sigmoid(),Linear(10,5),Sigmoid(),Linear(5,10),Softmax()],CrossEntropyLoss())

epoch = 16

for i in range(epoch):
    for X,Y in tqdm(batches):
        y_pred = model.forward(X)
        loss = model.compute_cost(Y, y_pred)
        model.backward()
        model.step()
    print("Epoch: ", i + 1, "Loss: ", loss)