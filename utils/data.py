import kaggle
import zipfile
import os
import shutil
import pandas as pd
import numpy as np
import math
from transforms import Transform

""" This Module is responsiple for 
1- Downloading ** a Kaggle dataset** and loading it into the model,
2- split trainig dataset into a train and validat pandas dataframes
3- load the dataframes to numpy array
4- seperate label from features 
5- reshape data to (WHC,N)
6- Normalize the dataset

**USAGE**
1- Create instance of the class 
2- Download the dataset using DataLoaderdownload_dataset(dataset_name) or simply add your data to folder named dataset 
    (note: the data **must** be in two files named trainig.csv and test.csv)
3- Call DataLoader.split_data(ratio) to split the data into validation and trainig ang intialize all member variables
4- use the get methods to get and needed data  
"""

class DataLoader:
    """
    Downloads data from internet and load it to data frame 
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

        train_data, test_data = self.__load_data(dataset_name)
        
        X_train, y_train, X_val, y_val = self.__split_data(train_data, self.split_ratio, self.shuffle)

        X_train, y_train, X_val, y_val = self.transform.to_tensor(X_train), \
                                         self.transform.to_tensor(y_train), \
                                         self.transform.to_tensor(X_val), \
                                         self.transform.to_tensor(y_val)
        
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val

        self.X_test = self.transform.to_tensor(test_data)


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
            shutil.rmtree(self.dataset_name)
        else:
            raise NameError("This dataset doesn't exits")

    
    # Load data from .csv files
    def __load_data(self, dataset_name):
        try:
            df = pd.read_csv(dataset_name + '/train.csv')
            df_test = pd.read_csv( dataset_name + '/test.csv')
        except:
            raise OSError("Wrong referred paths for data loading")

        return df, df_test

    # Split data randomly into train, validation sets
    def __split_data(self, train_data, split_ratio =0.2, shuffle=False):
        
        """
            Split trainig data which into train and validation dataframes by a split ratio (default 0.2)
        """
        df = train_data
        
        if shuffle is True:
            df_train = df.sample(frac=1-split_ratio)
            df_validation = df.drop(df_train.index)
        else:
            pass

        X_train = df_train.iloc[:, 1:] 
        y_train = df_train.iloc[:, 0]

        X_val = df_validation.iloc[:, 1:]
        y_val = df_validation.iloc[:, 0]
        
        return X_train, y_train, X_val, y_val
    

    def __partition(self, X, Y):
        m = X.shape[0]
        mini_batches = []
        num_mini_batches= math.floor(m / self.batch_size)
        print("Enterd Part")
        for i in range(0, num_mini_batches):
            mini_batch_X = X[i*self.batch_size:(i+1)*self.batch_size , :]
            mini_batch_Y = Y[i*self.batch_size:(i+1)*self.batch_size]

            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)

        # Handling the end case (last mini-batch < mini_batch_size)
        if m % self.batch_size != 0:
            mini_batch_X = X[self.batch_size*num_mini_batches:, :]
            mini_batch_Y = Y[self.batch_size*num_mini_batches:]

            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)

        return mini_batches

    def get_train_data(self):
        """returns train dataset as array and the train label as a vector"""
        return self.X_train, self.y_train

    def get_validation_data(self):
        """returns validation dataset as array and the validation label as a vector"""
        return self.X_val, self.y_val

    def get_test_data(self):
        """returns test dataset as array"""
        return self.X_test
    
    def get_train_sample(self, index):
        """returns the row with the specific index"""
        return self.X_train[index], self.y_train[index]
    
    def get_val_sample(self, index):
        """returns the row with the specific index"""
        return self.X_val[index], self.y_val[index]

    def get_test_sample(self, index):
        """returns the row with the specific index"""
        return self.X_test[index]

    def get_batched_data(self, X, Y):
        if self.batch_size != 1:
            return self.__partition(X, Y)
        else:
            return self.X_train, self.y_train
