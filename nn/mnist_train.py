from tqdm import tqdm

from utils.data import DataLoader
from nn.model import Model
from nn.activations import *
from nn.layers import *
from nn.losses import *


data_loader = DataLoader(batch_size=64)
X_train, y_train = data_loader.get_train_data()

y_train = y_train.reshape(1,y_train.shape[0])
X_train = X_train.reshape(28, 28, 1,X_train.shape[-1])
batches = data_loader.get_batched_data(X_train, y_train)

#model = Model([Linear(X_train.shape[0],10),Sigmoid(),Linear(10,10),Sigmoid()],CrossEntropyLoss())
model = Model([Conv2D(1,4),Sigmoid(),MaxPool2D(),Flatten(),Softmax()],CrossEntropyLoss())
epoch = 16

for i in range(epoch):
    for X,Y in tqdm(batches):
        print(X.shape)
        y_pred = model.forward(X[:, :, :, 0:64])
        print(y_pred)
        loss = model.compute_cost(Y[0, 0:64].T, y_pred)
        #model.backward()
        #model.step()
    print("Epoch: ", i + 1, "Loss: ", loss)