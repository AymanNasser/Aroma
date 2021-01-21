from data import DataLoader
import pandas as pd
import os
import sys

sys.path.insert(1, os.getcwd())


data_loader = DataLoader(dataset_path=os.getcwd() + '/nn/', split_ratio=0.2, download=False, batch_size=64, shuffle=True) 

X_train, y_train = data_loader.get_train_data()
dataset = data_loader.get_batched_data(X_train, y_train)
print(X_train.shape, y_train.shape)
print(len(dataset))