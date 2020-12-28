# Modules Definition & Design Issues

## Data Module

Dataset download link: <a href="https://www.kaggle.com/c/digit-recognizer/data" target="_top">Dataset</a>

### Data Downloading & Loading 
Using pandas, numpy, os & request packages

1. Inspect dataset description from kaggle
2. Download the dataset as a .csv file from a specified link </br>
3. Create a specific folders for training, validation & testing
4. Split **randomly** the training .csv file into training & validation dataset with a `SPLIT_RATIO = 0.2` 
5. Load an example by index from the dataset

### Data Transforming

1. Transform the specified .csv data to numpy arrays </br> 
    - Transformation is applied to **training, validation & testing** sets 
    - Separate the labels from the image data
2. Transform (reshape) the data shape according to working strategy:
    - Standard layer -> flatten the input to shape **(WxHxC,N)** 
    - CNN layer -> transform the shape to (N,C,W,H)
        - W: image width
        - H: image height
        - C: image channels, 1- for __grey scale__ & 3- for __RGB__
        - N: no. of samples of the set
    
3. Normalize the **training & validation** sets 
4. Transform a specific example for evaluation process 
    - When we predict an example, we need to transform this example to numpy array & normalize it for forward passing it through the model
5. Optional, Batching the training set into m-batches

Hints: 
- <a href="https://stackoverflow.com/questions/49386920/download-kaggle-dataset-by-using-python" target="_top">How to download a dataset from kaggle</a>
- <a href="https://stackoverflow.com/questions/43697240/how-can-i-split-a-dataset-from-a-csv-file-for-training-and-testing" target="_top">How to split .csv data</a>
- <a href="https://stackoverflow.com/questions/4315506/load-csv-into-2d-matrix-with-numpy-for-plotting" target="_top">How to load .csv data to numpy multi-dim array</a>
- <a href="https://stackoverflow.com/questions/1735025/how-to-normalize-a-numpy-array-to-within-a-certain-range" target="_top">How to normalize a numpy array</a>

## Evaluation Module

1. Predict a specific input through forward passing 
    - Pick the largest probability from the outputs 
2. Implement accuracy estimation function as it takes:
    - Predictions & ground truth labels
    - Equation may be in this form: `accuracy = float((np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100)`    
5. Optional, Implement precision & recall metric for better evaluation

## Visualization Module
Using matplotlib & pillow packages

1. Visualize a sample by index from the dataset
    - Image sample is given as **WxH** one-dim row so, we've to reshape the sample to visualize it
2. Visualize the cost function versus ***iterations/epochs*** during training process
    - Cost function is a 1D-Array
    - Specify the x,y & title names
3. Visualize the **accuracy** for training process

## Core Module
This module consits of mainly the implementation of 4-Parts:

- Layers
- Losses
- Activation functions
- Optimization algorithms

## Utility Module
This module is for saving & loading model weights & configurations into a compressed format 
