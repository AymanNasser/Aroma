from model import Model
from layers import *
from activations import *
from losses import *

import numpy as np

X = np.array([[1],[2],[3]])
print(X.shape)

model = Model([Linear(3,10),Sigmoid(),Linear(10,1),Sigmoid()],MSELoss())

result = model.forward(X)
print("FINAL LAYER",result)
loss = model.compute_cost(np.array([[0.5]]),result)

# S_vector = S.reshape(S.shape[0], 1)
# S_matrix = np.tile(S_vector, S.shape[0])
# S_dir = np.diag(S) - (S_matrix * np.transpose(S_matrix))
        

model.backward()
# model.step()

print("LOSS",result,loss)

