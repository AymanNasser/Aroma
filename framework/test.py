from model import Model
from layers import *
from activations import *
from losses import *
import numpy as np

model_name = "mnist"

X = np.array([[1],[2],[3]])
model = Model([Linear(3,10),Sigmoid(),Linear(10,1),Sigmoid()],MSELoss(),model_name=model_name)
for i in range(50):
    result = model.forward(X)
    loss = model.compute_cost(np.array([[0.5]]),result)
    model.backward()
    model.step()

model.save_model()
file_name = "%s_%s.pa"% (model_name,str(model.get_weights_number()))
model.load_model(file_name)