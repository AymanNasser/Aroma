import sys
import os
from tqdm import tqdm

sys.path.insert(1, str(os.getcwd()) )
sys.path.insert(1, str(os.getcwd()) + '/utils')

from utils.data import DataLoader
from model import Model
from activations import *
from layers import *
from losses import *


data_loader = DataLoader(batch_size=64)
X_train, y_train = data_loader.get_train_data()

y_train = y_train.reshape(y_train.shape[0],1)
batches = data_loader.get_batched_data(X_train, y_train)

model = Model([Linear(X_train.shape[0],10),Sigmoid(),Linear(10,10),Sigmoid()],CrossEntropyLoss())

epoch = 16

for i in range(epoch):
    for X,Y in tqdm(batches):
        y_pred = model.forward(X)
        loss = model.compute_cost(Y, y_pred)
        model.backward()
        model.step()