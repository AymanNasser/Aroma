import os, sys
sys.path.insert(1, os.getcwd())

from nn.model import Model
from nn.activations import *
from nn.layers import *
from nn.losses import *
from optim.optimizers import Adam, SGD
from utils.data import DataLoader
from eval.evaluation import Evaluation
from utils.transforms import Transform
from tqdm import tqdm
from matplotlib import pyplot as plt

INPUT_FEATURE = 784

data_loader = DataLoader(str(os.getcwd()) + '/nn',batch_size=16)
# data_loader = DataLoader(batch_size=64)

# Training
X_train, y_train = data_loader.get_train_data()
trans = Transform()
X_train = trans.normalize(X_train)
batches = data_loader.get_batched_data(X_train, y_train)


# Validation
X_val, Y_val = data_loader.get_validation_data()
X_val = trans.normalize(X_val)



model = Model([Linear(INPUT_FEATURE,128, init_type='xavier'),
               ReLU(),
               Linear(128,64, init_type='xavier'),
               ReLU(),
               Linear(64,32, init_type='xavier'),
               ReLU(),
               Linear(32,16, init_type='xavier'),
               ReLU(),
               Linear(16,10, init_type='xavier'),
               Softmax()], NLLLoss(), SGD(lr=0.001))

# model = Model([Conv2D(1,4),Sigmoid(),MaxPool2D(),Flatten(),Linear(676,10),Softmax()],CrossEntropyLoss())
# model = Model([Conv2D(1,4),Sigmoid(),Flatten(),Linear(2704,10),Softmax()],CrossEntropyLoss())
epoch = 0
print(model.get_count_model_params())

# model.load_model(os.getcwd() + '/model_111514.pa')

cost = 0.
for i in range(epoch):
    for X,Y in tqdm(batches):
        y_pred = model.forward(X)
        loss = model.compute_cost(Y, y_pred)
        model.backward()
        model.step()
    print("Epoch: ", i + 1, "Loss: ", loss)
    cost += loss

# print("Average Cost: ", cost / len(batches))
# model.save_model()

# Evaulating model
Pred_ = model.predict(X_val)
Pred_ = np.argmax(Pred_, axis=0)
Y_val = Y_val.T.squeeze()

eval = Evaluation(Y_val, Pred_)
acc = eval.compute_accuracy()
prec = eval.compute_precision()
recall = eval.compute_recall()
f1_score = eval.compute_f1_score()
print("Accuracy: ",acc,"Precision: ",prec,"Recall: ",recall,"F1_Score: ",f1_score)   

