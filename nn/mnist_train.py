from tqdm import tqdm

from utils.data import DataLoader
from nn.model import Model
from nn.activations import *
from nn.layers import *
from nn.losses import *

data_loader = DataLoader(batch_size=64)
X_train, y_train = data_loader.get_train_data(tensor_shape='4D',H=28,W=28,C=1)
batches = data_loader.get_batched_data(X_train, y_train)

# model = Model([Linear(X_train.shape[0],20),Sigmoid(),Linear(20,15),Sigmoid(),
#                Linear(15,10),Softmax()],CrossEntropyLoss())
model = Model([Conv2D(1,4),Sigmoid(),MaxPool2D(),Flatten(),Softmax()],CrossEntropyLoss())
epoch = 16

for i in range(epoch):
    for X,Y in tqdm(batches):
        y_pred = model.forward(X)
        loss = model.compute_cost(Y, y_pred)
        print("Epoch: ", i + 1, "Loss: ", loss)
        # model.backward()
        # model.step(learning_rate=0.1)