import tensorflow as tf
from segmented_data_builder import tf_Data_Builder
import os
import sys
import matplotlib.pyplot as plt
import numpy as np

data_dir = '/Users/lukeprice/github/multi-modal/datafiles/Y8M_segmented/'
trainFiles = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if '.tfrecord' in file]

tfd = tf_Data_Builder(
  batchsize = 32, 
  target_classes = np.array([[170],[1454],[709],[1057],[1308]]))
IDs, labels, images, audio = tfd.create_dataset(tf_files = trainFiles)

