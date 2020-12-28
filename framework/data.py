import kaggle
import zipfile
import os
import shutil
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


data_ldr = DataLoader()
data_ldr.delete_dataset()
