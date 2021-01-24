import os,sys
sys.path.insert(1, os.getcwd())

from dataloader import DataLoader
import pandas as pd




data_loader = DataLoader(dataset_path=os.getcwd() + '/nn/', split_ratio=0.2, download=False, batch_size=64, shuffle=True) 

df = data_loader.get_pandas_frame()
print(df.describe())

X_train, y_train = data_loader.get_train_data(tensor_shape='2D', H=28, W=28, C=1)
X_val, y_val = data_loader.get_validation_data(tensor_shape='4D', H=28, W=28, C=1)
X_test = data_loader.get_test_data(tensor_shape='4D', H=28, W=28, C=1)

dataset = data_loader.get_batched_data(X_train, y_train)
print(X_train.shape, y_train.shape)
print(X_val.shape, y_val.shape)
print(X_test.shape)

print(dataset[0][0].shape, dataset[0][1].shape)
RR = data_loader.get_train_sample(1)
print(RR[0].shape, RR[1].shape)