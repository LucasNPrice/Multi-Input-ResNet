import tensorflow as tf
from tqdm import tqdm
import os 
import sys
import numpy as np
from sklearn.preprocessing import OneHotEncoder

""" NOTE:
    audio samples seem to be different lengths
    - the tails of each audio sample is all zero 
    - zeros begin at 244, None, 176, for example """

class tf_Data_Builder():

  def __init__(self, batchsize, target_classes):
    self.batchsize = batchsize
    self.feature_num = 300
    self.classes = target_classes
    self.__fit_embeddings()

  def __fit_embeddings(self):
    """ fit one-hot encoder """
    self.onehot_encoder = OneHotEncoder(sparse=False, categories='auto')
    self.onehot_encoder.fit(self.classes)

  def create_dataset(self, tf_files):
    """ create batched dataset to feed to network """
    dataset = tf.data.TFRecordDataset(tf_files)
    dataset = dataset.map(self.__parse_function, num_parallel_calls=4)
    dataset = dataset.repeat()    
    dataset = dataset.shuffle(len(tf_files))
    dataset = dataset.batch(self.batchsize)
    
    iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
    IDs, labels, images, audio = iterator.get_next()
    labels = tf.sparse.to_dense(labels)
    labels = self.multi_hot_classes(labels)

    images = tf.sparse.to_dense(images)
    images = tf.io.decode_raw(
      input_bytes = images, 
      out_type = tf.uint8, 
      fixed_length = 1024)
    images = tf.reshape(images, 
      [self.batchsize, 
      self.feature_num, 
      32, 
      32])
    audio = tf.sparse.to_dense(audio)
    audio = tf.io.decode_raw(
      input_bytes = audio, 
      out_type = tf.uint8, 
      fixed_length = 128)
    audio = tf.reshape(audio, 
      [self.batchsize, self.feature_num, tf.shape(audio)[-1]])
    # audio = tf.reshape(audio, 
    #   [self.batchsize, self.feature_num*128])

    return IDs, labels, images, audio

  def __parse_function(self, raw_tfrecord):
    """ create dictionary structure for aquiring .tfrecord data """
    context_features = {
        'id': tf.io.FixedLenFeature([], dtype=tf.string),
        'labels': tf.io.VarLenFeature(dtype=tf.int64)
    }
    sequence_features = {
        'audio': tf.io.VarLenFeature(dtype=tf.string),
        'rgb': tf.io.VarLenFeature(dtype=tf.string)
    }
    context_data, sequence_data = tf.io.parse_single_sequence_example(
      serialized = raw_tfrecord,
      context_features = context_features,
      sequence_features = sequence_features)

    IDs = context_data['id']
    labels = context_data['labels']
    audio = sequence_data['audio']
    images = sequence_data['rgb']

    return IDs, labels, images, audio

  def multi_hot_classes(self, labels):
    """ 
    transforms a tensor of variable length labels to a multi-hot array of labels 
    labels: label array to transform
    returns: multi-hot array representation of labels 
    """
    labels = np.array(labels)
    onehot_labels = []
    for i, label in enumerate(labels):
      encoded_label = np.zeros(len(self.classes))
      for j, val in enumerate(label):
        if val != 0:
          new_val = self.onehot_encoder.transform(val.reshape(1,-1))
          encoded_label += new_val[0]
      onehot_labels.append(encoded_label)
    # print(tf.convert_to_tensor(onehot_labels, tf.int64))

    return np.array(onehot_labels)




  