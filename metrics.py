import tensorflow as tf
import numpy as np
from sklearn.metrics import multilabel_confusion_matrix

class Metrics():

  def __init__(self, y_true, y_pred):

    """ Creates object of class Metrics """

    self.y_true = y_true
    self.y_pred = y_pred
    self.conf_mat = multilabel_confusion_matrix(self.y_true, 
                                                self.y_pred)
    conf_mat_sum = np.zeros((2,2))
    for mat in self.conf_mat:
      conf_mat_sum += mat
    self.true_negatives, self.false_positives, self.false_negatives, self.true_positives = conf_mat_sum.flatten()

  def get_precision(self, return_metric = False):

    self.precision = self.true_positives / (self.true_positives + self.false_positives)
    if return_metric:
      return self.precision

  def get_recall(self, return_metric = False):

    self.recall = self.true_positives / (self.true_positives + self.false_negatives)
    if return_metric:
      return self.recall

  def get_F1(self, return_metric = False):
    
    precision = self.get_precision(True)
    recall = self.get_recall(True)
    self.f1_score = 2 * ((precision * recall) / (precision + recall))
    if return_metric:
      return self.f1_score

if __name__ == '__main__':
  # multi_modal_weights was written over
  # multi_modal_metrics was incorrectly written two extra metrics 
  import matplotlib.pyplot as plt
  import pickle
  multi_metric_file = '/Users/lukeprice/github/multi-modal/metric_files/multi_modal_metrics.pickle'
  image_metric_file = '/Users/lukeprice/github/multi-modal/metric_files/image_only_metrics.pickle'
  with open(metric_file, 'rb') as file:
    logged_metrics = pickle.load(file)
  f1_history = []
  Epochs = logged_metrics['Epochs']
  for epoch in Epochs:
    print(Epochs[epoch]['metrics'])
    f1_history.append(Epochs[epoch]['metrics']['F1'])
  print(len(f1_history))
  # plt.plot(f1_history)
  # plt.show()