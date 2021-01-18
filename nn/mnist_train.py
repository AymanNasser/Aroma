import sys
import os
sys.path.insert(1, str(os.getcwd()) )
sys.path.insert(1, str(os.getcwd()) + '/utils')


from utils.data import DataLoader
from model import Model
from activations import *
from layers import *
from losses import *
import numpy as np



data_loader = DataLoader('/home/ayman/FOE-Linux/Aroma', batch_size=64)
X_train, y_train = data_loader.get_train_data()
# batches = data_loader.get_batched_data(X_train, y_train)

model = Model([Linear(X_train.shape[1],10),Sigmoid(),Linear(10,1),Sigmoid()],MSELoss())

epoch = 8
j = 0
print(X_train.shape)
for i in range(epoch):

    y_pred = model.forward(X_train[j:j+1000, :])
    loss = model.compute_cost(y_train[j:j+1000], y_pred)
    model.backward()
    model.step()
    j += 1000
    print(i)

print(loss)
print(y_pred)

