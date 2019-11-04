import tensorflow as tf
from segmented_data_builder import tf_Data_Builder
import os
import sys
import matplotlib.pyplot as plt
import numpy as np

data_dir = '/Users/lukeprice/github/multi-modal/datafiles/Y8M_segmented/'
trainFiles = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if '.tfrecord' in file]

# tfd = tf_Data_Builder(
#   tf_files = trainFiles,
#   batchsize = 32, 
#   target_classes = np.array([[170],[1454],[709],[1057],[1308]]))
# IDs, frames, labels, images, audio = tfd.create_dataset()

def __parse_function(raw_tfrecord):

  features = {
    'id': tf.io.FixedLenFeature([],dtype=tf.string),
    'labels': tf.io.VarLenFeature(dtype=tf.int64),
    'frame': tf.io.FixedLenFeature([],dtype=tf.int64),
    'audio': tf.io.FixedLenFeature([],dtype=tf.string),
    'rgb': tf.io.FixedLenFeature([],dtype=tf.string)
  }
  data = tf.io.parse_single_example(
    serialized = raw_tfrecord,
    features = features)
  # IDs = data['id']
  # frame = data['frame']
  # labels = data['labels']
  # audio = data['audio']
  # images = data['rgb']
  # return images, labels
  return data 

dataset = tf.data.TFRecordDataset(trainFiles)
dataset = dataset.map(__parse_function, num_parallel_calls=4)
dataset = dataset.repeat(10)    
dataset = dataset.shuffle(len(trainFiles))
dataset = dataset.batch(32, drop_remainder=True)
for x in dataset:
  print(x)
  input()
# iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
# next_element = iterator.get_next()


# with tf.compat.v1.Session() as sess:
#   sess.run(tf.compat.v1.global_variables_initializer())
#   try:
#     for i in range(10):
#     # while True:
#       next_batch = sess.run(next_element)
#       print(next_batch)
#       input()
#   except:
#     pass


