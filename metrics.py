import tensorflow as tf
import numpy as np
from sklearn.metrics import multilabel_confusion_matrix

class Metrics():
  def __init__(self, y_true, y_pred):
    self.y_true = y_true
    self.y_pred = y_pred
    self.conf_mat = multilabel_confusion_matrix(
        self.y_true, 
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
