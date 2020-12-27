# Modules Definition 

## Data Module

Dataset download link: <a href="https://www.kaggle.com/c/digit-recognizer/data" target="_top">Dataset</a>

### Data Downloading & Loading 

1. Inspect dataset description from kaggle
2. Download the dataset as a .csv file from a specified link </br>
3. Create a specific folders for training, validation & testing
4. Split **randomly** the training .csv file into training & validation dataset with a `SPLIT_RATIO = 0.2` 

### Data Transforming

1. Transform the specified .csv data to numpy arrays </br> 
    - Transformation is applied to **training, validation & testing** sets 
    - Separate the labels from the image data
2. Transform (reshape) the data shape according to working strategy:
    - Standard layer -> flatten the input to shape **(W*H*C, N)** 
    - CNN layer -> transform the shape to (N,C,W,H)
        - W: image width
        - H: image height
        - C: image channels, 1- for __grey scale__ & 3- for __RGB__
        - N: no. of samples of the set
    
3. Normalize the **training & validation** sets 

Hints: 
- <a href="https://stackoverflow.com/questions/49386920/download-kaggle-dataset-by-using-python" target="_top">How to download a dataset from kaggle</a>
- <a href="https://stackoverflow.com/questions/43697240/how-can-i-split-a-dataset-from-a-csv-file-for-training-and-testing" target="_top">How to split .csv data</a>
- <a href="https://stackoverflow.com/questions/4315506/load-csv-into-2d-matrix-with-numpy-for-plotting" target="_top">How to load .csv data to numpy multi-dim array</a>
- <a href="https://stackoverflow.com/questions/1735025/how-to-normalize-a-numpy-array-to-within-a-certain-range" target="_top">How to normalize a numpy array</a>

## Evaluation Module


## Visualization Module