import tensorflow as tf
import keras.backend as K
import numpy as np

class FocalLoss():
  def __init__(self, alpha, gamma = 2.0, class_proportions = True): 
    if class_proportions:
      self.alpha = 1-alpha
    else:
      self.alpha = alpha
    self.gamma = gamma

  def __call__(self, y_true, y_pred):
    y_pred = tf.clip_by_value(
      t=y_pred, 
      clip_value_min=0+K.epsilon(), 
      clip_value_max=1-K.epsilon())
    # determine where target is 0
    zeros = tf.equal(y_true, 0)
    # get p_t: change p_t = 1-p if y = 0 
    p_t = tf.where(zeros, 1-y_pred, y_pred)
    # get modulating factor (1-p_t)^gamma
    modulating_factor = tf.pow(1-p_t,self.gamma)
    # compute focal loss
    focal_loss = -self.alpha * modulating_factor * tf.math.log(p_t)
    return K.mean(K.sum(focal_loss, axis = 1))
