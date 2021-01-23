import os, sys
sys.path.insert(1, os.getcwd())

from nn.model import Model
from nn.activations import *
from nn.layers import *
from nn.losses import *
from optim.adam import Adam
from utils.data import DataLoader
from eval.evaluation import Evaluation
from utils.transforms import Transform
from tqdm import tqdm
import pandas as pd

df = pd.read_csv(os.getcwd() + '/Data.csv')
X = df.iloc[:, :-1]
Y = df.iloc[:, -1]
print(df.describe())

X = X.to_numpy().T
Y = Y.to_numpy()

X = (X - np.mean(X)) / np.std(X)
Y = Y.reshape(1,Y.shape[0])
print(X.shape, Y.shape)

model = Model([Linear(X.shape[0],128, init_type='xavier'),
               ReLU(),
               Linear(128,64, init_type='xavier'),
               ReLU(),
               Linear(64,32, init_type='xavier'),
               ReLU(),
               Linear(32,1, init_type='xavier'),
               ReLU()], MSELoss(), Adam(lr=0.01))


epoch = 64

for i in range(epoch):
    y_pred = model.forward(X)
    loss = model.compute_cost(Y, y_pred)
    model.backward()
    model.step()
    print("Epoch: ", i + 1, "Loss: ", loss)
