import tensorflow as tf
from tqdm import tqdm
import os 
import sys
import numpy as np
from sklearn.preprocessing import OneHotEncoder

class tf_Data_Builder():

  def __init__(self, tf_files, batchsize, target_classes):
    self.tfrecords = tf_files
    self.dataset = tf.data.TFRecordDataset(self.tfrecords)
    self.batchsize = batchsize
    # self.feature_num = 300
    self.classes = target_classes
    self.__fit_embeddings()
    # self.compile_model()
  
  def __fit_embeddings(self):
    # """ """
    # labs = tf.feature_column.categorical_column_with_vocabulary_list(
    #   'labs', [170, 1454, 709, 1057, 1308])
    # self.labs_one_hot = tf.feature_column.indicator_column(labs)
    # print(self.labs_one_hot)
    # input()
    """ fit one-hot encoder """
    self.onehot_encoder = OneHotEncoder(sparse=False, categories='auto')
    self.onehot_encoder.fit(self.classes)

  def create_dataset(self):
    """ create batched dataset to feed to network """
    # dataset = tf.data.TFRecordDataset(tf_files)
    self.dataset = self.dataset.map(self.__parse_function, num_parallel_calls=4)
    self.dataset = self.dataset.repeat()    
    self.dataset = self.dataset.shuffle(len(self.tfrecords))
    self.dataset = self.dataset.batch(self.batchsize, drop_remainder=True)

    print(self.dataset)
    input()

    # iterator = tf.compat.v1.data.make_one_shot_iterator(self.dataset)
    # IDs, frames, labels, images, audio = iterator.get_next()

    # grade = tf.feature_column.categorical_column_with_vocabulary_list(
    #   'grade', ['poor', 'average', 'good'])
    # grade_one_hot = tf.feature_column.indicator_column(grade)

    # print(grade)
    # input()
    # print(grade_one_hot)
    # input()
    # data = {
    # 'marks': [55,21,63,88,74,54,95,41,84,52],
    # 'grade': ['average','poor','average','good','good','average','good','average','good','average'],
    # 'point': ['c','f','c+','b+','b','c','a','d+','b+','c']
    # }
    # from tensorflow.keras import layers
    # def demo(feature_column):
    #   feature_layer = layers.DenseFeatures(feature_column)
    #   print(feature_layer)
    #   input()
    #   print(feature_layer(data).numpy())
    # demo(grade_one_hot)

    # for l in labels:
    #   print(l)
    # input()
    # from tensorflow.keras import layers
    # def demo(feature_column):
    #   feature_layer = layers.DenseFeatures(feature_column)
    #   print(feature_layer(labels).numpy())
    # demo(self.labs_one_hot)

    # labels = tf.sparse.to_dense(labels)
    # labels = self.multi_hot_classes(labels)

    # images = tf.io.decode_raw(
    #   input_bytes = images, 
    #   out_type = tf.uint8)
    # images = tf.reshape(images, 
    #   [self.batchsize, 32, 32, 1])
    # audio = tf.io.decode_raw(
    #   input_bytes = audio, 
    #   out_type = tf.uint8)#, 

    # # return IDs, frames, labels, images, audio
    # return images, labels
    # return self.dataset

  def __parse_function(self, raw_tfrecord):

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
    labels = tf.sparse.to_dense(labels)
    # labels = tf.one_hot(labels, 171)
    # labels = tf.reshape(labels, [171])

    # thal = feature_column.categorical_column_with_vocabulary_list(
    #   'thal', ['fixed', 'normal', 'reversible'])

    # labs = tf.feature_column.categorical_column_with_vocabulary_list(
    #   'labs', [170, 1454, 709, 1057, 1308])
    # labs_one_hot = tf.feature_column.indicator_column(labs)
    # demo(labs_one_hot)
    # input()
    # labels = self.multi_hot_classes(labels)

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

    return images, labels
    return IDs, frame, labels, images, audio

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

    return np.array(onehot_labels)

  def compile_model(self):
    from tensorflow.keras import optimizers
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, Activation, Flatten, Dense, MaxPooling2D, Dropout, BatchNormalization
    from tensorflow import optimizers
    optimizer = optimizers.Adam(lr=0.0005)
    self.model = Sequential()
    self.model.add(Conv2D(input_shape = (32, 32, 1),
      filters = 64, 
      kernel_size = 3,
      activation = 'relu'))
    self.model.add(BatchNormalization())
    self.model.add(MaxPooling2D(pool_size=(2, 2)))
    self.model.add(Dropout(0.2))
    self.model.add(Flatten())
    self.model.add(Dense(171, activation = 'sigmoid'))
    self.model.summary()
    self.model.compile(optimizer = optimizer, 
      loss=tf.keras.losses.binary_crossentropy, 
      metrics=['accuracy'])

if __name__ == '__main__':

  data_dir = '/Users/lukeprice/github/multi-modal/datafiles/Y8M_segmented/'
  trainFiles = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if '.tfrecord' in file]
  tfd = tf_Data_Builder(
    tf_files = trainFiles,
    batchsize = 32, 
    target_classes = np.array([[170],[1454],[709],[1057],[1308]]))
  tfd.create_dataset()
   

