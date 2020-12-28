import kaggle
import zipfile
import os
import shutil
import pandas as pd
import numpy as np

""" This Module is responsiple for 
1- Downloading ** a Kaggle dataset** and loading it into the model,
2- split trainig dataset into a train and validat pandas dataframes
3- load the dataframes to numpy array
4- seperat label from features 
5- reshape data to (WHC,N) or (W,H,C,N)
6- Normalize the dataset
"""


class DataLoader:
    """
    Downloads data from internet and load it to data frame 
    """
    def __init__(self):
        pass

    def download_dataset(self, dataset_name):
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
    def split_data(self , ratio = 0.2 ):
        """split trainig data which is in Aroma/framework/dataset/train.csv
            into train and validation dataframes by a ratio (default 0.2)
            and return train and validation dataframes respectively
        """
        df = pd.read_csv('./dataset/train.csv')
        df['split'] = np.random.randn(df.shape[0], 1)

        msk = np.random.rand(len(df)) <= ratio

        train = df[~msk]
        validation = df[msk]
        return train,validation

