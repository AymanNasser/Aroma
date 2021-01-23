from tqdm import tqdm

from utils.data import DataLoader
from nn.model import Model
from nn.activations import *
from nn.layers import *
from nn.losses import *
from utils.transforms import Transform

data_loader = DataLoader(batch_size=128)
trans = Transform()
# X_train, y_train = data_loader.get_train_data(tensor_shape='4D',H=28,W=28,C=1)
X_train, y_train = data_loader.get_train_data()
X_train = trans.normalize(X_train)
batches = data_loader.get_batched_data(X_train, y_train)
x_val, y_val = data_loader.get_validation_data()

model = Model([Linear(X_train.shape[0],128,'zero'),Sigmoid(),Linear(128,64,'xavier'),Sigmoid(),
                Linear(64,32,'xavier'),Sigmoid(),Linear(32,10,'xavier'),Softmax()],CrossEntropyLoss())
# model = Model([Conv2D(1,4),Sigmoid(),MaxPool2D(),Flatten(),Linear(676,10),Softmax()],CrossEntropyLoss())
# model = Model([Conv2D(1,4),Sigmoid(),Flatten(),Linear(2704,10),Softmax()],CrossEntropyLoss())
epoch = 16

for i in range(epoch):
    for X,Y in tqdm(batches):
        y_pred = model.forward(X)
        loss = model.compute_cost(Y, y_pred)
        model.backward()
        model.step(learning_rate=0.1)
    print("Epoch: ", i + 1, "Loss: ", loss)

# model.forward(x_val)