import tensorflow as tf
from tqdm import tqdm
import os 
import sys
import numpy as np
from sklearn.preprocessing import OneHotEncoder

class tf_Data_Builder():

  def __init__(self):
    pass

  def create_dataset(self, tf_datafiles, batch_size):
    """ creates Dataset object to feed to network """
    self.dataset = tf.data.TFRecordDataset(tf_datafiles)
    self.dataset = self.dataset.map(self.__parse_function, num_parallel_calls=4)
    # self.dataset = self.dataset.repeat()    
    self.dataset = self.dataset.shuffle(len(tf_datafiles))
    self.dataset = self.dataset.batch(batch_size, drop_remainder=True)

  def __parse_function(self, raw_tfrecord):
    """ builds data structure transformation pipeline """
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
    IDs = data['id']
    frame = data['frame']
    labels = data['labels']
    # labels = tf.sparse.to_dense(labels)
    audio = data['audio']
    audio = tf.io.decode_raw(
      input_bytes = audio, 
      out_type = tf.uint8)
    images = data['rgb']
    img_dim = 32
    images = tf.io.decode_raw(
      input_bytes = images, 
      out_type = tf.uint8)
    images = tf.reshape(images, [img_dim, img_dim, 1])
    images = tf.image.per_image_standardization(images)

    return images, labels
    return IDs, frame, labels, images, audio

  def fit_multi_hot_encoder(self, class_labels):
    """ fits one-hot encoder """
    self.class_labels = class_labels
    self.onehot_encoder = OneHotEncoder(sparse=False, categories='auto')
    self.onehot_encoder.fit(self.class_labels)

  def multi_hot_classes(self, labels):
    """ 
    transforms a tensor of variable length labels to a multi-hot array of labels 
    labels: label array to transform
    returns: multi-hot array representation of labels 
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

if __name__ == '__main__':

  data_dir = '/Users/lukeprice/github/multi-modal/datafiles/Y8M_segmented/'
  trainFiles = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if '.tfrecord' in file]
  data_builder = tf_Data_Builder()
  data_builder.create_dataset(tf_datafiles = trainFiles, batch_size = 32)
  for batch, (image, label) in enumerate(data_builder.dataset):
    print(image[0])
    input()
   

