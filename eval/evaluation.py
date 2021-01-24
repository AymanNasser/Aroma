import numpy as np


class Evaluation:
    def __init__(self, Y_true, Y_pred, average='macro'):
        assert Y_pred.shape == Y_pred.shape
        
        self.__Y_true = Y_true                                  
        self.__Y_pred = Y_pred  
        self.__TP = None
        self.__FP = None
        self.__TN = None
        self.__FN = None
        self.__average = average
        self.__samples_per_class = [] 

        self.__conf_mat = self.compute_confusion_mat() 
        self.__calc_class_samples()
    
    def __calc_class_samples(self):
        num_classes = len(np.unique(self.__Y_true))
        for i in range(num_classes):
            self.__samples_per_class.append(np.sum(self.__Y_true == i))

    def compute_confusion_mat(self):
        num_classes = len(np.unique(self.__Y_true))
        conf_mat = np.zeros((num_classes, num_classes))

        for i in range(len(self.__Y_true)):
            conf_mat[self.__Y_true[i]][self.__Y_pred[i]] += 1

        self.__FP = conf_mat.sum(axis=0) - np.diag(conf_mat)  
        self.__FN = conf_mat.sum(axis=1) - np.diag(conf_mat)
        self.__TP = np.diag(conf_mat)
        self.__TN = conf_mat.sum() - (self.__FP + self.__FN + self.__TP)

        return conf_mat


    def compute_accuracy(self): 
        acc = (self.__TP + self.__TN) / (self.__TP + self.__TN + self.__FP + self.__FN)
        if self.__average == 'macro':
            acc = np.mean(acc)
        
        elif self.__average == 'weighted':
            acc = np.sum(np.multiply(self.__samples_per_class, acc)) / len(self.__Y_true)
        
        else:
            raise AttributeError("Wrong averaging attribute")
    
        return acc * 100

    def compute_recall(self):
        recall = self.__TP / (self.__TP + self.__TN) 
        if self.__average == 'macro':
            recall = np.mean(recall) 
        
        elif self.__average == 'weighted':
                recall = np.sum(np.multiply(self.__samples_per_class, recall)) / len(self.__Y_true)
        
        else:
            raise AttributeError("Wrong averaging attribute")
    
        return recall * 100

    def compute_precision(self):
        precision = self.__TP / (self.__TP + self.__FN)
        if self.__average == 'macro':
            precision = np.mean(precision)

        elif self.__average == 'weighted':
            precision = np.sum(np.multiply(self.__samples_per_class, precision)) / len(self.__Y_true)
        
        else:
            raise AttributeError("Wrong averaging attribute")
    
        return precision * 100

    def compute_f1_score(self):
        f1_score = (2*self.__TP) / (2*self.__TP + self.__FP + self.__FN)
        if self.__average == 'macro':
            f1_score = np.mean(f1_score)

        elif self.__average == 'weighted':
            f1_score = np.sum(np.multiply(np.array(self.__samples_per_class), f1_score)) / len(self.__Y_true)

        else:
            raise AttributeError("Wrong averaging attribute")

        return f1_score * 100
