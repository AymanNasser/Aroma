import kaggle
import zipfile
import os
from shutil import rmtree
import pandas as pd
from math import floor
from utils.transforms import Transform
import numpy as np


class DataLoader:
    """
        Data module is responsiple for 
            1- Download ** a Kaggle dataset** and loading it into a dataframe
            2- Split dataset into train., valid. and testing dataframes
            3- Reshaping data to (N x M) or (H x W x C x M)
            4- Partitioning data into m-batches
    """  
    def __init__(self, dataset_path=os.getcwd(), dataset_name='digit-recognizer', split_ratio=0.2, download=False, batch_size=1, shuffle=False):
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.download_dataset = download
        self.batch_size = batch_size
        self.shuffle = shuffle 
        self.split_ratio = split_ratio
        self.transform = Transform()

        if download is True:
            self.__download_dataset(dataset_name)
        else:
            pass

        train_data, test_data = self.__load_data()
        self.pd_frame = train_data

        X_train, y_train, X_val, y_val = self.__split_data(train_data, self.split_ratio, self.shuffle)
        
        X_train, y_train, X_val, y_val = self.transform.to_tensor(X_train), \
                                         self.transform.to_tensor(y_train), \
                                         self.transform.to_tensor(X_val), \
                                         self.transform.to_tensor(y_val)
        
        self.X_train = X_train.reshape(X_train.shape[0], -1).T
        self.y_train = y_train.reshape(y_train.shape[0], -1).T
        self.X_val = X_val.reshape(X_val.shape[0], -1).T
        self.y_val = y_val.reshape(y_val.shape[0], -1).T

        self.X_test = self.transform.to_tensor(test_data)
        self.X_test = self.X_test.reshape(self.X_test.shape[0], -1).T

    def __download_dataset(self, dataset_name):
        """takes a dataset name and download it from **kaggle**, unzip it and remove the zip file"""

        kaggle.api.authenticate()
        kaggle.api.competition_download_files(dataset_name, path=self.dataset_path)

        filename = dataset_name + ".zip"

        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall('./' + str(dataset_name))

        os.remove(filename)

    def delete_dataset(self, dataset_name):
        """Removes the downloaded data"""
        if os.path.exists(dataset_name):
            rmtree(self.dataset_name)
        else:
            raise NameError("This dataset doesn't exits")

    # Load data from .csv files
    def __load_data(self):
        try:
            print(self.dataset_path + '/' + self.dataset_name +  '/train.csv')
            df = pd.read_csv(self.dataset_path + '/' + self.dataset_name +  '/train.csv')
            df_test = pd.read_csv(self.dataset_path + '/' + self.dataset_name +  '/test.csv')
        except:
            raise OSError("Wrong referred paths for data loading")

        return df, df_test

    # Split data randomly into train, validation sets
    def __split_data(self, train_data, split_ratio =0.2, shuffle=False):
        
        """
            Split trainig data which into train and validation dataframes by a split ratio (default 0.2)
        """
        df = train_data
                
        df_train = df.sample(frac=1-split_ratio)
        df_validation = df.drop(df_train.index)

        X_train = df_train.iloc[:, 1:] 
        y_train = df_train.iloc[:, 0]

        X_val = df_validation.iloc[:, 1:]
        y_val = df_validation.iloc[:, 0]
        
        return X_train, y_train, X_val, y_val
    
    def __reshape_data(self, df):
        pass


    def __partition(self, X, Y):
        assert len(X.shape) == 2 or len(X.shape) == 4, "Unsupported tensor shape for batching"
        
        m = X.shape[-1]
        mini_batches = []
        num_mini_batches= floor(m / self.batch_size)
        
        # 2D Tensor
        if len(X.shape) == 2:
            for i in range(0, num_mini_batches):
                mini_batch_X = X[:, i*self.batch_size:(i+1)*self.batch_size]
                mini_batch_Y = Y[:, i*self.batch_size:(i+1)*self.batch_size]

                mini_batch = (mini_batch_X, mini_batch_Y)
                mini_batches.append(mini_batch)

            # Handling the end case (last mini-batch < mini_batch_size)
            if m % self.batch_size != 0:
                mini_batch_X = X[:, self.batch_size*num_mini_batches:]
                mini_batch_Y = Y[:, self.batch_size*num_mini_batches:]

                mini_batch = (mini_batch_X, mini_batch_Y)
                mini_batches.append(mini_batch)

        # 4D Tensor
        elif len(X.shape) == 4:
            for i in range(0, num_mini_batches):
                mini_batch_X = X[:, :, :, i*self.batch_size:(i+1)*self.batch_size]
                mini_batch_Y = Y[:, i*self.batch_size:(i+1)*self.batch_size]

                mini_batch = (mini_batch_X, mini_batch_Y)
                mini_batches.append(mini_batch)

            # Handling the end case (last mini-batch < mini_batch_size)
            if m % self.batch_size != 0:
                mini_batch_X = X[:, :, :, self.batch_size*num_mini_batches:]
                mini_batch_Y = Y[:, self.batch_size*num_mini_batches:]

                mini_batch = (mini_batch_X, mini_batch_Y)
                mini_batches.append(mini_batch)

        else:
            pass

        return mini_batches

    def get_train_data(self, tensor_shape='2D', H=None, W=None, C=None):
        """returns train dataset as array and the train label as a vector"""
        if tensor_shape == '2D':
            return self.X_train, self.y_train

        elif tensor_shape == '4D':
            assert H is not None or W is not None or C is not None
            M = self.X_train.shape[-1]
            X_train = self.X_train.reshape(H, W, C, M)
            return X_train, self.y_train

        else:
            raise AttributeError("Wrong tensor shape, select either 2D or 4D")

    def get_validation_data(self, tensor_shape='2D', H=None, W=None, C=None):
        """returns validation dataset as array and the validation label as a vector"""
        if tensor_shape == '2D':
            return self.X_val, self.y_val

        elif tensor_shape == '4D':
            assert H is not None or W is not None or C is not None
            M = self.X_val.shape[-1]
            X_val = self.X_val.reshape(H, W, C, M)
            return X_val, self.y_train

        else:
            raise AttributeError("Wrong tensor shape, select either 2D or 4D")

    def get_test_data(self, tensor_shape='2D', H=None, W=None, C=None):
        """returns test dataset as array"""
        if tensor_shape == '2D':
            return self.X_test  

        elif tensor_shape == '4D':
            assert H is not None or W is not None or C is not None
            M = self.X_test.shape[-1]
            X_test = self.X_test.reshape(H, W, C, M)
            return X_test

        else:
            raise AttributeError("Wrong tensor shape, select either 2D or 4D")
    
    def get_train_sample(self, index):
        """returns the row with the specific index"""
        return self.X_train[:, index], self.y_train[:, index]
    
    def get_val_sample(self, index):
        """returns the row with the specific index"""
        return self.X_val[:, index], self.y_val[:, index]

    def get_test_sample(self, index):
        """returns the row with the specific index"""
        return self.X_test[:, index]

    def get_batched_data(self, X, Y):
        if self.batch_size != 1:
            return self.__partition(X, Y)
        else:
            return self.X_train, self.y_train

    def get_pandas_frame(self):
        return self.pd_frame