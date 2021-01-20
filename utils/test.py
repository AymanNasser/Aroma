from data import DataLoader


data_loader = DataLoader('/home/ayman/FOE-Linux/Aroma',split_ratio=0.2, download=False, batch_size=64, shuffle=True) 

X_train, y_train = data_loader.get_train_data()
dataset = data_loader.get_batched_data(X_train, y_train)
print(X_train.shape, y_train.shape)
print(len(dataset))