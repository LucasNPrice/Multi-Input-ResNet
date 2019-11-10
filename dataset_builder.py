import tensorflow as tf
from tqdm import tqdm
import os 
import sys
import numpy as np
from sklearn.preprocessing import OneHotEncoder

class tfDataBuilder():

  def __init__(self, mode):
    if mode == 'segmented':
      self.mode = 'segmented'
    elif mode == 'time_distributed':
      self.mode = 'time_distributed'
    else:
      raise NameError('mode must be either {} or {}'.format('\'segmented\'', '\'time_distributed\''))

  def create_single_dataset(self, tf_datafiles, batch_size):

    """ Creates Dataset object to feed to network """

    self.batch_size = batch_size
    self.dataset = tf.data.TFRecordDataset(tf_datafiles)

    self.dataset = self.dataset.map(self.__parse_function, num_parallel_calls=4)  

    # if self.mode == 'segmented':
    #   self.dataset = self.dataset.map(self.__segmented_parse_function, num_parallel_calls=4)  
    # elif self.mode == 'time_distributed':
    #   self.dataset = self.dataset.map(self.__time_distributed_parse_function, num_parallel_calls=4)  

    self.dataset = self.dataset.shuffle(len(tf_datafiles))
    self.dataset = self.dataset.batch(self.batch_size, drop_remainder=True)
    self.dataset_size = self.get_dataset_size(self.dataset)

  def create_train_test_dataset(self, train_tf_datafiles, test_tf_datafiles, batch_size):

    """ Creates a train and test Dataset object to feed to network """

    self.batch_size = batch_size
    self.train_dataset = tf.data.TFRecordDataset(train_tf_datafiles)
    self.test_dataset = tf.data.TFRecordDataset(test_tf_datafiles)

    self.train_dataset = self.train_dataset.map(self.__parse_function, num_parallel_calls=4)
    self.test_dataset = self.test_dataset.map(self.__parse_function, num_parallel_calls=4)

    # elif self.mode == 'time_distributed':
    # if self.mode == 'segmented':
    #   self.train_dataset = self.train_dataset.map(self.__segmented_parse_function, num_parallel_calls=4)
    #   self.test_dataset = self.test_dataset.map(self.__segmented_parse_function, num_parallel_calls=4)
    # elif self.mode == 'time_distributed':
    #   self.train_dataset = self.train_dataset.map(self.__time_distributed_parse_function, num_parallel_calls=4)
    #   self.test_dataset = self.test_dataset.map(self.__segmented_parse_function, num_parallel_calls=4)

    self.train_dataset = self.train_dataset.shuffle(len(train_tf_datafiles))
    self.test_dataset = self.test_dataset.shuffle(len(test_tf_datafiles))
    self.train_dataset = self.train_dataset.batch(self.batch_size, drop_remainder=True)
    self.test_dataset = self.test_dataset.batch(self.batch_size, drop_remainder=False)

    self.train_size = self.get_dataset_size(self.train_dataset)
    self.test_size = self.get_dataset_size(self.test_dataset)

  def __parse_function(self, raw_tfrecord):

    """ Builds data structure transformation pipeline """

    if self.mode == 'segmented':

      features = {
        'id': tf.io.FixedLenFeature([],dtype=tf.string),
        'labels': tf.io.VarLenFeature(dtype=tf.int64),
        'frame': tf.io.FixedLenFeature([],dtype=tf.int64),
        'audio': tf.io.FixedLenFeature([],dtype=tf.string),
        'rgb': tf.io.FixedLenFeature([],dtype=tf.string)
      }

      data = tf.io.parse_single_example(serialized=raw_tfrecord,
                                        features=features)

      IDs = data['id']
      frame = data['frame']
      labels = data['labels']
      # labels = tf.sparse.to_dense(labels)
      audio = data['audio']
      audio = tf.io.decode_raw(input_bytes=audio, 
                               out_type=tf.uint8)
      audio = tf.reshape(audio, [128,1])    
      images = data['rgb']
      img_dim = 32
      images = tf.io.decode_raw(input_bytes=images, 
                                out_type=tf.uint8)    
      images = tf.reshape(images, [img_dim, img_dim, 1])
      images = tf.image.per_image_standardization(images)

    elif self.mode == 'time_distributed':

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
      # audio = tf.sparse.to_dense(audio)
      # audio = tf.io.decode_raw(input_bytes=audio, 
      #                          out_type=tf.uint8,
      #                          fixed_length=128)
      images = sequence_data['rgb']
      # images = tf.sparse.to_dense(images)
      # images = tf.io.decode_raw(input_bytes=images, 
      #                           out_type=tf.uint8,
      #                           fixed_length=1024)  

    return images, audio, labels

  def fit_multi_hot_encoder(self, class_labels):

    """ Fits one-hot encoder """

    self.class_labels = class_labels
    self.onehot_encoder = OneHotEncoder(sparse=False, categories='auto')
    self.onehot_encoder.fit(self.class_labels)

  def multi_hot_classes(self, labels):

    """ Transforms a tensor of variable length labels to a multi-hot array of labels.
        labels: label array to transform.
        returns: multi-hot array representation of labels.
    """

    labels = np.array(labels)
    onehot_labels = []
    for i, label in enumerate(labels):
      encoded_label = np.zeros(len(self.class_labels))
      for j, val in enumerate(label):
        if val != 0:
          new_val = self.onehot_encoder.transform(val.reshape(1,-1))
          encoded_label += new_val[0]
      onehot_labels.append(encoded_label)

    return tf.cast(onehot_labels, tf.float64)

  def get_dataset_size(self, dataset):

    """ Returns the number of examples in a Dataset object """
    
    i = 0
    for batch, (image, audio, label) in enumerate(dataset):
      assert tf.shape(image)[0] == tf.shape(audio)[0] == tf.shape(label)[0]
      i += tf.shape(image)[0]
      
    return i


if __name__ == '__main__':

  data_dir = '/Users/lukeprice/github/multi-modal/datafiles/Y8M_build/'
  trainFiles = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if '.tfrecord' in file]
  data_builder = tfDataBuilder(mode='time_distributed')
  data_builder.create_single_dataset(tf_datafiles=trainFiles, batch_size=32)
  records = 0
  for batch, (image, audio, label) in enumerate(data_builder.dataset):
    image = tf.sparse.to_dense(image)
    image = tf.io.decode_raw(input_bytes=image, 
                          out_type=tf.uint8,
                          fixed_length=1024)
    for i in enumerate(image):
      records += 1
  print(records)
   

