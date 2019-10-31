import tensorflow as tf
import os
from segmented_data_builder import tf_Data_Builder
import numpy as np
from tensorflow import optimizers
import sys

""" 29545 records sharded across 41 .tfrecord files! """
""" Current Issue: batching labels of different sizes :
    sometimes in has 1 label and is 1D, other times it has 2 labels and 
    is 2D. """

def train_gradient_tape(epochs, dataset):
  for epoch in range(epochs):
    print('Start of epoch %d' % (epoch,))
    input()
    for batch, (image, label) in enumerate(dataset):
      print(batch)

if __name__ == '__main__':
  
  data_dir = '/Users/lukeprice/github/multi-modal/datafiles/Y8M_segmented/'
  trainFiles = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if '.tfrecord' in file]
  data_builder = tf_Data_Builder(
    tf_files = trainFiles,
    batchsize = 32, 
    target_classes = np.array([[170],[1454],[709],[1057],[1308]]))
  data_builder.create_dataset()
  train_gradient_tape(2, data_builder.dataset)
   


