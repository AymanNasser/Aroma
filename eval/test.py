import sys
import os
sys.path.insert(1, os.getcwd() )

from evaluation import Evaluation
from viz.visualization import Visualization
import numpy as np


y_true = np.array([1, -1,  0,  0,  1, -1,  1,  0, -1,  0,  1, -1,  1,  0,  0, -1,  0]) + 1
y_prediction = np.array([-1, -1,  1,  0,  0,  0,  0, -1,  1, -1,  1,  1,  0,  0,  1,  1, -1]) + 1
# y_prediction = np.array([1, -1,  0,  0,  1, -1,  1,  0, -1,  0,  1, -1,  1,  0,  0, -1,  0]) + 1



eval = Evaluation(y_true, y_prediction, average='weighted')
print(eval.compute_confusion_mat())

print(eval.compute_accuracy())
print(eval.compute_recall())
print(eval.compute_precision())
print(eval.compute_f1_score())

Viz = Visualization()
Viz.plot_confusion_matrix(eval.compute_confusion_mat())