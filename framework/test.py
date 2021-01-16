from model import Model
from layers import *
from activations import *
from losses import *

import numpy as np

X = np.array([[1],[2],[3]])
print(X.shape)

model = Model([Linear(3,2),ReLU(),Linear(2,1),Sigmoid()],MSELoss())

result = model.forward(X)
loss = model.compute_cost(np.array([[0.5]]),np.array([[0.5]]))

model.backward()
model.step()

print(result,loss)

