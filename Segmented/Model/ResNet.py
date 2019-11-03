import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras import layers, Input, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.layers import Add, ZeroPadding2D, Flatten, Dense
from keras.activations import relu
from segmented_data_builder import tf_Data_Builder
from tqdm import tqdm
import os
import numpy as np
import sys
import unittest

class ResNet():
  def __init__(self, trim_front = False, trim_end = False,  **kwargs):
    self.trim_front = trim_front
    self.trim_end = trim_end
    # if True, we dont need shape 
    # if false, we need shape 
    if trim_front:
      self.X_input = kwargs.get('X_input')
    else:
      assert kwargs.get('input_shape') is not None
      self.input_shape = kwargs.get('input_shape')
      self.X_input = Input(self.input_shape)
    if not trim_end:
      assert kwargs.get('num_classes') is not None
      self.num_classes = kwargs.get('num_classes')


  def identity_block(self, X, filters, kernel_size, stage, block):
    """ filters: list of 3 ints defining number of filters in each Conv2d layer 
        kernel_size: int defining the kernel_size of the middle Conv2d layer 
        stage: int to name/describe which stage in the total network """
    X_shortcut = X
    F1, F2, F3 = filters
    ks = (kernel_size, kernel_size)
    conv_name = 'Conv_Stage_' + str(stage) + '_' + str(block)
    BN_name = 'BN_Stage_' + str(stage) + '_' + str(block)

    # first block
    X = Conv2D(filters = F1, kernel_size = (1,1), strides = (1,1), padding = 'valid', 
      name = conv_name + '_a')(X)
    X = BatchNormalization(axis = -1, name = BN_name + '_a')(X)
    X = relu(X)
    
    # middle block 
    X = Conv2D(filters = F2, kernel_size = ks, strides = (1,1), padding = 'same', 
      name = conv_name + '_b')(X)
    X = BatchNormalization(axis = -1, name = BN_name + '_b')(X)
    X = relu(X)

    # last block
    X = Conv2D(filters = F3, kernel_size = (1,1), strides = (1,1), padding = 'valid', 
      name = conv_name + '_c')(X)
    X = BatchNormalization(axis = -1, name = BN_name + '_c')(X)

    X = Add()([X, X_shortcut])
    X = relu(X)
    return X

  def convolutional_block(self, X, filters, kernel_size, stage, block, stride = 2):
    """ filters: list of 3 ints defining number of filters in each Conv2d layer 
    kernel_size: int defining the kernel_size of the middle Conv2d layer 
    stage: int to name/describe which stage in the total network """
    X_shortcut = X
    F1, F2, F3 = filters
    ks = (kernel_size, kernel_size)
    conv_name = 'Conv_Stage_' + str(stage) + '_' + str(block)
    BN_name = 'BN_Stage_' + str(stage) + '_' + str(block)
    s = stride

    # first block
    X = Conv2D(filters = F1, kernel_size = (1,1), strides = (s,s), padding = 'valid', 
      name = conv_name + '_a')(X)
    X = BatchNormalization(axis = -1, name = BN_name + '_a')(X)
    X = relu(X)

    # middle block 
    X = Conv2D(filters = F2, kernel_size = ks, strides = (1,1), padding = 'same', 
      name = conv_name + '_b')(X)
    X = BatchNormalization(axis = -1, name = BN_name + '_b')(X)
    X = relu(X)

    # last block
    X = Conv2D(filters = F3, kernel_size = (1,1), strides = (1,1), padding = 'valid', 
      name = conv_name + '_c')(X)
    X = BatchNormalization(axis = -1, name = BN_name + '_c')(X)

    # shortcut path - ensure equal outut dimensions (F3)
    X_shortcut = Conv2D(filters = F3, kernel_size = (1, 1), strides = (s,s), padding = 'valid', 
      name = conv_name + '_shortcut')(X_shortcut)
    X_shortcut = BatchNormalization(axis = -1, name = BN_name + '_shortcut')(X_shortcut)

    X = Add()([X, X_shortcut])
    X = relu(X)
    return X


  # def resnet(self, input_shape = (32,32,1), classes = 5, save_img = False):
  def resnet(self):

    """ Put together structure and depth of model here by 
        stacking identity and convolutional layers """
    # if self.trim_front:
    #   X_input = X_input
    X = ZeroPadding2D((3, 3))(self.X_input)

    # stage 1
    X = Conv2D(filters = 64, 
      kernel_size = (7, 7), 
      strides=(2, 2), 
      name = 'Conv_Stage_1')(X)
    X = BatchNormalization(name = 'BN_Stage_1')(X)
    X = relu(X)
    X = MaxPooling2D(pool_size = (3, 3), strides = (2, 2))(X)

    # stage 2
    X = self.convolutional_block(X = X, 
      filters=[64, 64, 256], 
      kernel_size=3, 
      stage=2, stride=1, block = 'block_A')
    X = self.identity_block(X = X, 
      filters = [64, 64, 256], 
      kernel_size = 3, stage = 2, block = 'block_B')
    X = self.identity_block(X = X, 
      filters = [64, 64, 256], 
      kernel_size = 3, stage = 2, block = 'block_C')

    if self.trim_end: 
      return X
    else:
      # output layers
      X = Flatten()(X)
      X = Dense(units = self.num_classes, activation = 'sigmoid', name = 'final_dense')(X)
      
      # create model
      model = Model(inputs = self.X_input, outputs = X, name='myResNet')
      keras.utils.plot_model(model, 'ResNet.png')
      return model


if __name__ == '__main__':
  
  # set data path and files 
  data_dir = '/Users/lukeprice/github/multi-modal/datafiles/Y8M_segmented/'
  trainFiles = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if '.tfrecord' in file]

  # build dataset
  data_builder = tf_Data_Builder()
  data_builder.create_dataset(
    tf_datafiles = trainFiles, 
    batch_size = 32)
  data_builder.fit_multi_hot_encoder(
    class_labels = np.array([[170],[1454],[709],[1057],[1308]]))

  resnet = ResNet(input_shape = (32,32,1), num_classes = 5)
  model = resnet.resnet()
  

  # create model object plus train, predict, and evaluate
  # model_object = Multi_Modal(data_builder)
  # model_object.compile_multi_modal_network(5, True, True)
  # model_object.train_model(1)
  # model_object.predict_model()
  # model_object.get_model_metrics()




