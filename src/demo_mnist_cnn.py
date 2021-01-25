from nn.model import Model
from nn.activations import *
from nn.layers import *
from nn.model import Model
from nn.activations import *
from nn.layers import *
from nn.losses import *
from optim.optimizers import Adam, SGD
from utils.dataloader import DataLoader
from eval.evaluation import Evaluation
from viz.visualization import Visualization
from utils.transforms import Transform
from utils.process_tensor import padding
from tqdm import tqdm

data_loader = DataLoader(batch_size=64, dataset_path="../")

# Training
X_train, y_train = data_loader.get_train_data(tensor_shape='4D', H=28, W=28, C=1)
trans = Transform()
X_train = trans.normalize(X_train)

# Reshaping tensor to 32*32*1
X_train = padding(X_train, (2,2))

# Batching data
batches = data_loader.get_batched_data(X_train, y_train)

# Validation
X_val, Y_val = data_loader.get_validation_data()
X_val = trans.normalize(X_val)

INPUT_FEATURE = 784
momentum = 0.9
learning_rate = 0.001
EPOCHS = 16

# Training LeNet-5
model = Model(layers=[Conv2D(in_channels=1, out_channels=6, kernel_size=(5,5), stride=(1,1), padding=(0,0), init_type='random'),
                    Tanh(),
                    AvgPool2D(kernel_size=2, stride=2, padding=0),
             
                    Conv2D(in_channels=6, out_channels=16, kernel_size=(5,5), stride=(1,1), padding=(0,0), init_type='random'), 
                    Tanh(), 
                    AvgPool2D(kernel_size=(2,2), stride=(2,2), padding=0),
             
                    Flatten(),
                    Linear(400,120),
                    Tanh(),
                    Linear(120,84),
                    Tanh(),
                    Softmax()], loss=NLLLoss(), optimizer=SGD(lr=learning_rate, momentum=momentum), model_name='CNN_Model')
epoch = 16

vis = Visualization()

for i in range(epoch):
    for X,Y in tqdm(batches):
        y_pred = model.forward(X)
        loss = model.compute_cost(Y, y_pred)
        model.backward()
        model.step()
    vis.plot_live_update(xlabel="Epoch No.", x=i + 1, ylabel="Loss", y=loss)

vis.pause_figure()

model.save_model()

# Evaluating model
Pred_ = model.predict(X_val)
Pred_ = np.argmax(Pred_, axis=0)
Y_val = Y_val.T.squeeze()

eval = Evaluation(Y_val, Pred_, average='weighted')
acc = eval.compute_accuracy()
prec = eval.compute_precision()
recall = eval.compute_recall()
f1_score = eval.compute_f1_score()
conf_mat = eval.compute_confusion_mat()
print("Accuracy: ",acc,"Precision: ",prec,"Recall: ",recall,"F1_Score: ",f1_score)

vis.plot_confusion_matrix(conf_mat)
