from nn.model import Model
from nn.activations import *
from nn.layers import *
from nn.losses import *
from optim.adam import Adam
from utils.data import DataLoader
from eval.evaluation import Evaluation
from utils.transforms import Transform
from tqdm import tqdm

# import os, sys
# sys.path.insert(1, os.getcwd())
# data_loader = DataLoader(str(os.getcwd()) + '/nn',batch_size=32)

data_loader = DataLoader(batch_size=64)

# X_train, y_train = data_loader.get_train_data(tensor_shape='4D',H=28,W=28,C=1)
X_train, y_train = data_loader.get_train_data()
trans = Transform()
X_train = trans.normalize(X_train)

batches = data_loader.get_batched_data(X_train, y_train)
x_val, y_val = data_loader.get_validation_data()
x_val = trans.normalize(x_val)

model = Model([Linear(X_train.shape[0],128, init_type='random'),
               ReLU(),
               Linear(128,64, init_type='random'),
               ReLU(),
               Linear(64,32, init_type='random'),
               ReLU(),
               Linear(32,10, init_type='random'),
               Softmax()], NLLLoss(), Adam(lr=0.01))

# model = Model([Conv2D(1,4),Sigmoid(),MaxPool2D(),Flatten(),Linear(676,10),Softmax()],CrossEntropyLoss())
# model = Model([Conv2D(1,4),Sigmoid(),Flatten(),Linear(2704,10),Softmax()],CrossEntropyLoss())
epoch = 16

for i in range(epoch):
    for X,Y in tqdm(batches):
        y_pred = model.forward(X)
        loss = model.compute_cost(Y, y_pred)
        model.backward()
        model.step()
    print("Epoch: ", i + 1, "Loss: ", loss)

eval = Evaluation(Y,y_pred)
acc = eval.compute_accuracy()
prec = eval.compute_precision()
recall = eval.compute_recall()
f1_score = eval.compute_f1_score()
print("Accuracy: ",acc,"Precision: ",prec,"Recall: ",recall,"F1_Score: ",f1_score)