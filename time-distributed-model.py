import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras import layers, Input, Model
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import concatenate
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import load_model
from tqdm import tqdm
import os
import numpy as np
import sys
import copy

from ResNet import ResNet
from dataset_builder import tfDataBuilder
from focal_loss import FocalLoss
from metrics import Metrics


class TimeDistributedMultiModal():

  def __init__(self, data_builder_object):

    """ Creates an object of class MultiModal.

    Must be called with tfDataBuilder object containng a tensorflow Dataset object. 

    Args:
      data_builder_object: Object of class tf_Data_Builder. 
      pretrained_model: (Optional) pretrained multi-input model in .h5 format.  
      json_model: (Optional) model defined in .json format. 
      weights: (Optional) weights to assign to model form .json. 
    """

    self.data_builder = data_builder_object
    self.batch_size = self.data_builder.batch_size

  def preprocess(self, images, audio):

    """ Preprocess image and audio data for network processing. """

    images = tf.sparse.to_dense(images)
    images = tf.io.decode_raw(input_bytes=images, 
                              out_type=tf.uint8,
                              fixed_length=1024)
    images = tf.cast(images, tf.float32)

    audio = tf.sparse.to_dense(audio)
    audio = tf.io.decode_raw(input_bytes=audio, 
                              out_type=tf.uint8,
                              fixed_length=128)
    audio = tf.cast(audio, tf.float32)

    return images, audio

  def train_model(self, epochs):
    
    for epoch in range(epochs):
      num_batches = int(self.data_builder.dataset_size//self.batch_size)
      with tqdm(total=num_batches) as pbar:
        for batch, (images, audio, labels) in enumerate(self.data_builder.dataset):
          images, audio = self.preprocess(images, audio)
          print(images)
          input()
          print(audio)
          sys.exit()
          pbar.update(1)

if __name__ == '__main__':

  # set data path and files 
  train_dir = '/Users/lukeprice/github/multi-modal/datafiles/Y8M_build/'
  trainFiles = [os.path.join(train_dir, file) for file in os.listdir(train_dir) if '.tfrecord' in file]

  # build dataset (currently only classifying 5 labels)
  data_builder = tfDataBuilder(mode='time_distributed')
  data_builder.fit_multi_hot_encoder(class_labels=np.array([[170],[1454],[709],[1057],[1308]]))
  data_builder.create_single_dataset(tf_datafiles=trainFiles,  
                                     batch_size=32)  

  model = TimeDistributedMultiModal(data_builder_object=data_builder)
  model.train_model(epochs=3)



