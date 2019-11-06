import tensorflow as tf
import keras.backend as K
# from tensorflow import math
# from tensorflow import keras 
import numpy as np

class FocalLoss():
  def __init__(self, alpha, gamma = 2.0, class_proportions = True): 
    if class_proportions:
      self.alpha = 1-alpha
    else:
      self.alpha = alpha
    self.gamma = 2.0

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

# def focal_loss(self, target, y_hat):#, alpha, gamma = 2.0, class_proportions = True):
#   """ 
#   y = Ground Truth value (true label)
#   if y = 1, p_t = p, alpha_t = alpha
#   else if y = 0 (!=1), pt = 1-p, alpha_t = 1-alpha
#   form: -alpha_t * (1-p_t)^gamma * log(p_t) 

#   alpha = vector of weighting values; typically (inverse) of class proportions 
#   class_proportions: if True, alpha = 1 - alpha such that majority classes are down-weighted more than minority classes
#   """
#   # if class_proportions:
#   #   alpha = 1-alpha
#   # gamma = 2.0
#   # clip out 0 and 1 values from y_hat for compatibility with log()
#   y_hat = tf.clip_by_value(t=y_hat, 
#     clip_value_min=0+K.epsilon(), 
#     clip_value_max=1-K.epsilon())
#   # determine where target is 0
#   zeros = tf.equal(target, 0)
#   # get p_t: change p_t = 1-p if y = 0 
#   p_t = tf.where(zeros, 1-y_hat, y_hat)
#   # get modulating factor (1-p_t)^gamma
#   modulating_factor = tf.pow(1-p_t,self.gamma)
#   # compute focal loss
#   focal_loss = -self.alpha * modulating_factor * tf.math.log(p_t)
#   return K.mean(K.sum(focal_loss, axis = 1))

# if __name__ == '__main__':

#   # 1 = good, 2 = bad
#   predicted = np.array([[.9,.3,.8,.2,.1], [.9,.8,.5,.7,.1]])
#   true_labels = np.array([[1,0,1,0,0], [1,0,0,0,1]])

#   focal_loss(target = true_labels, y_hat = predicted)





  print('running well')