import kaggle
import zipfile
import os
import shutil
from numpy.lib.shape_base import split
import pandas as pd
import numpy as np

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
    
    def __init__(self):
        pass

    def download_dataset(self, dataset_name = 'digit-recognizer'):
        """takes a dataset name and download it from **kaggle**, unzip it and remove the zip file"""

        kaggle.api.authenticate()

        kaggle.api.competition_download_files(dataset_name)

        filename = dataset_name + ".zip"
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall('./dataset')

        os.remove(filename)

    def delete_dataset(self):
        """Removes the downloaded data"""
        shutil.rmtree('dataset')

    
    
    #load data into model
    def split_data(self, ratio =0.2):
        
        """split trainig data which is in Aroma/framework/dataset/train.csv
            into train and validation dataframes by a ratio (default 0.2)
            and load test data which is in Aroma/framework/dataset/test.csv
            and intialize five member vaiables
            
        """
        df = pd.read_csv('./dataset/train.csv') 
        df_test = pd.read_csv('./dataset/test.csv')
        df['split'] = np.random.randn(df.shape[0], 1)

        msk = np.random.rand(len(df)) <= ratio

        df_train = df[~msk]
        df_validation = df[msk]

        self.train = df_train.iloc[:,1:-1].to_numpy()/255 
        self.train_label = df_train.loc[:,'label'].values
        self.validation = df_validation.iloc[:,1:-1].to_numpy()/255
        self.validation_label = df_validation.loc[:,'label'].values
        self.test = df_test.iloc[:,1:-1].to_numpy()
        #self.test_label = df_test.loc[:,'label'].values
    
    def get_train_data(self):
        """returns train dataset as array and the train label as a vector"""
        return self.train,self.train_label
    
    def get_validation_data(self):
        """returns validation dataset as array and the validation label as a vector"""
        return self.validation,self.validation_label

    def get_test_data(self):
        """returns test dataset as array"""
        return self.test
    
    def get_train_data(self, index ):
        """returns the row with the specific index"""
        return self.train[index],self.train_label[index]
    
    def get_validation_data(self,index):
        """returns the row with the specific index"""
        return self.validation[index],self.validation_label[index]

    def get_test_data(self,index):
        """returns the row with the specific index"""
        return self.test[index]

