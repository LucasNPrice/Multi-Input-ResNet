import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras import layers, Input, Model
from tensorflow.keras.layers import ZeroPadding2D, Conv2D, MaxPooling2D
from tensorflow.keras.layers import Add, Flatten, Dense, BatchNormalization
from tensorflow.keras.layers import ZeroPadding1D, Conv1D, MaxPooling1D
from keras.activations import relu
from tqdm import tqdm
import os
import numpy as np
import sys

from segmented_data_builder import tfDataBuilder

class ResNet():

  def __init__(self, trim_front = False, trim_end = False,  **kwargs):

    """ Creates an object of class ResNet.

    Args:
      trim_front: True to remove input layer. 
                  False to keep input layer. 
                  True for transfer learning. 
      trim_end: True to remove final output (logit) layer (and return feature map). 
                False to keep final output layer (and return logits). 
                True for transfer learning. 
      input_shape: (Optional) shape of input data X.
      num_classes: (Optional) number of classes in softmax/sigmoid output layer.
    """

    self.trim_front = trim_front
    self.trim_end = trim_end
    if trim_front:
      self.X_input = kwargs.get('X_input')
    else:
      assert kwargs.get('input_shape') is not None
      self.input_shape = kwargs.get('input_shape')
      self.X_input = Input(self.input_shape)
    if not trim_end:
      assert kwargs.get('num_classes') is not None
      self.num_classes = kwargs.get('num_classes')

  def identity_block_1D(self, X, filters, kernel_size, stage, block):

    """ 
    Args:
      X: Input data/tensor. 
      filters: List of 3 ints defining number of filters in each Conv1D layer. 
      kernel_size: Int defining the kernel_size of the middle Conv1D layer.
      stage: Name of stage of blocks in the total network (a descriptor).
      block: Name of block within stage (a descriptor).
    """

    X_shortcut = X
    F1, F2, F3 = filters
    conv_name = 'Conv1D_Stage_' + str(stage) + '_Block_' + str(block)
    BN_name = 'BN1D_Stage_' + str(stage) + '_Block_' + str(block)

    # first block
    X = Conv1D(filters = F1, kernel_size = 1, strides = 1, padding = 'valid', 
               kernel_initializer='glorot_uniform', name = conv_name + '_a')(X)
    X = BatchNormalization(axis = -1, name = BN_name + '_a')(X)
    X = relu(X)

    # middle block 
    X = Conv1D(filters = F2, kernel_size = kernel_size, strides = 1, padding = 'same', 
               kernel_initializer='glorot_uniform', name = conv_name + '_b')(X)
    X = BatchNormalization(axis = -1, name = BN_name + '_b')(X)
    X = relu(X)

    # last block
    X = Conv1D(filters = F3, kernel_size = 1, strides = 1, padding = 'valid', 
               kernel_initializer='glorot_uniform', name = conv_name + '_c')(X)
    X = BatchNormalization(axis = -1, name = BN_name + '_c')(X)

    X = Add()([X, X_shortcut])
    X = relu(X)
    return X

  def convolutional_block_1D(self, X, filters, kernel_size, stage, block, stride = 2):
    
    """ 
    Args:
      X: Input data/tensor.
      filters: List of 3 ints defining number of filters in each Conv1D layer.
      kernel_size: Int defining the kernel_size of the middle Conv1D layer.
      stage: Name of stage of blocks in the total network (a descriptor).
      block: Name of block within stage (a descriptor).
    """

    X_shortcut = X
    F1, F2, F3 = filters
    conv_name = 'Conv1D_Stage_' + str(stage) + '_Block_' + str(block)
    BN_name = 'BN1D_Stage_' + str(stage) + '_Block_' + str(block)

    # first block
    X = Conv1D(filters = F1, kernel_size = 1, strides = stride, padding = 'valid', 
               kernel_initializer='glorot_uniform', name = conv_name + '_a')(X)
    X = BatchNormalization(axis = -1, name = BN_name + '_a')(X)
    X = relu(X)

    # middle block 
    X = Conv1D(filters = F2, kernel_size = kernel_size, strides = 1, padding = 'same', 
               kernel_initializer='glorot_uniform', name = conv_name + '_b')(X)
    X = BatchNormalization(axis = -1, name = BN_name + '_b')(X)
    X = relu(X)

    # last block
    X = Conv1D(filters = F3, kernel_size = 1, strides = 1, padding = 'valid', 
               kernel_initializer='glorot_uniform', name = conv_name + '_c')(X)
    X = BatchNormalization(axis = -1, name = BN_name + '_c')(X)

    # shortcut path - ensure equal outut dimensions (F3)
    X_shortcut = Conv1D(filters = F3, kernel_size = 1, strides = stride, padding = 'valid', 
                        kernel_initializer='glorot_uniform', name = conv_name + '_shortcut')(X_shortcut)
    X_shortcut = BatchNormalization(axis = -1, name = BN_name + '_shortcut')(X_shortcut)

    X = Add()([X, X_shortcut])
    X = relu(X)
    return X


  def ResNet1D(self, save_img=False):

    """ Builds the residual network architecture by calling identity_block_1D and convolutional_block_1D. 
        Puts together structure and depth of model here by stacking identity and convolutional layers.
        More stages/blocks may be added at will.
    """

    # X_input = X_input
    X = ZeroPadding1D(3)(self.X_input)

    # stage 1
    X = Conv1D(filters = 32, 
               kernel_size = 7, 
               strides=2, 
               kernel_initializer='glorot_uniform',
               name = 'Conv1D_Stage_1')(X)
    X = BatchNormalization(name = 'BN1D_Stage_1')(X)
    X = relu(X)
    X = MaxPooling1D(pool_size = 3, strides = 2)(X)

    # stage 2
    X = self.convolutional_block_1D(X = X, filters=[32, 32, 126], kernel_size=3, 
                                    stride=1, stage = 2, block = 'A')
    X = self.identity_block_1D(X = X, filters = [32, 32, 126], kernel_size = 3, 
                               stage = 2, block = 'B')
    X = self.identity_block_1D(X = X, filters = [32, 32, 126], kernel_size = 3, 
                               stage = 2, block = 'C')


    # stage 3
    X = self.convolutional_block_1D(X = X, filters = [64, 64, 256], kernel_size = 3, 
                                    stride = 2, stage = 3, block = 'A')
    X = self.identity_block_1D(X = X, filters = [64, 64, 256], kernel_size = 3, 
                               stage = 3, block = 'B')
    X = self.identity_block_1D(X = X, filters = [64, 64, 256], kernel_size = 3, 
                               stage = 3, block = 'C')    
    X = self.identity_block_1D(X = X, filters = [64, 64, 256], kernel_size = 3, 
                               stage = 3, block = 'D')

    # stage 4
    X = self.convolutional_block_1D(X = X, filters=[128, 128, 512], kernel_size=3, 
                                    stride = 2, stage = 4, block = 'A')
    X = self.identity_block_1D(X = X, filters = [128, 128, 512], kernel_size = 3, 
                               stage = 4, block = 'B')
    X = self.identity_block_1D(X = X, filters = [128, 128, 512], kernel_size = 3, 
                               stage = 4, block = 'C')    
    X = self.identity_block_1D(X = X, filters = [128, 128, 512], kernel_size = 3, 
                               stage = 4, block = 'D')
    X = self.identity_block_1D(X = X, filters = [128, 128, 512], kernel_size = 3, 
                               stage = 4, block = 'E')    
    X = self.identity_block_1D(X = X, filters = [128, 128, 512], kernel_size = 3, 
                               stage = 4, block = 'F')

    if self.trim_end: 
      # return feature map and do NOT perform sigmoid activation.
      return X
    else:
      # get output layers and perform sigmoid activation.
      X = Flatten()(X)
      X = Dense(units = self.num_classes, activation = 'sigmoid', name = 'final_dense')(X)
      
      # create model
      model = Model(inputs = self.X_input, outputs = X, name='myResNet')
      if save_img:
        keras.utils.plot_model(model, 'ResNet1D.png')
      return model

  def identity_block_2D(self, X, filters, kernel_size, stage, block):

    """ 
    Args:
      X: Input data/tensor.
      filters: List of 3 ints defining number of filters in each Conv2d layer.
      kernel_size: Int defining the kernel_size of the middle Conv2d layer.
      stage: Name of stage of blocks in the total network (a descriptor).
      block: Name of block within stage (a descriptor).
    """

    X_shortcut = X
    F1, F2, F3 = filters
    ks = (kernel_size, kernel_size)
    conv_name = 'Conv2D_Stage_' + str(stage) + '_Block_' + str(block)
    BN_name = 'BN2D_Stage_' + str(stage) + '_Block_' + str(block)

    # first block
    X = Conv2D(filters = F1, kernel_size = (1,1), strides = (1,1), padding = 'valid', 
               kernel_initializer='glorot_uniform', name = conv_name + '_a')(X)
    X = BatchNormalization(axis = -1, name = BN_name + '_a')(X)
    X = relu(X)
    
    # middle block 
    X = Conv2D(filters = F2, kernel_size = ks, strides = (1,1), padding = 'same', 
               kernel_initializer='glorot_uniform', name = conv_name + '_b')(X)
    X = BatchNormalization(axis = -1, name = BN_name + '_b')(X)
    X = relu(X)

    # last block
    X = Conv2D(filters = F3, kernel_size = (1,1), strides = (1,1), padding = 'valid', 
               kernel_initializer='glorot_uniform', name = conv_name + '_c')(X)
    X = BatchNormalization(axis = -1, name = BN_name + '_c')(X)

    X = Add()([X, X_shortcut])
    X = relu(X)
    return X

  def convolutional_block_2D(self, X, filters, kernel_size, stage, block, stride = 2):
   
    """ 
    Args:
      X: Input data/tensor.
      filters: List of 3 ints defining number of filters in each Conv2d layer.
      kernel_size: Int defining the kernel_size of the middle Conv2d layer.
      stage: Name of stage of blocks in the total network (a descriptor).
      block: Name of block within stage (a descriptor).
    """

    X_shortcut = X
    F1, F2, F3 = filters
    ks = (kernel_size, kernel_size)
    conv_name = 'Conv2D_Stage_' + str(stage) + '_Block_' + str(block)
    BN_name = 'BN2D_Stage_' + str(stage) + '_Block_' + str(block)
    s = stride

    # first block
    X = Conv2D(filters = F1, kernel_size = (1,1), strides = (s,s), padding = 'valid', 
               kernel_initializer='glorot_uniform', name = conv_name + '_a')(X)
    X = BatchNormalization(axis = -1, name = BN_name + '_a')(X)
    X = relu(X)

    # middle block 
    X = Conv2D(filters = F2, kernel_size = ks, strides = (1,1), padding = 'same', 
               kernel_initializer='glorot_uniform', name = conv_name + '_b')(X)
    X = BatchNormalization(axis = -1, name = BN_name + '_b')(X)
    X = relu(X)

    # last block
    X = Conv2D(filters = F3, kernel_size = (1,1), strides = (1,1), padding = 'valid', 
               kernel_initializer='glorot_uniform', name = conv_name + '_c')(X)
    X = BatchNormalization(axis = -1, name = BN_name + '_c')(X)

    # shortcut path - ensure equal outut dimensions (F3)
    X_shortcut = Conv2D(filters = F3, kernel_size = (1, 1), strides = (s,s), padding = 'valid', 
                        kernel_initializer='glorot_uniform', name = conv_name + '_shortcut')(X_shortcut)
    X_shortcut = BatchNormalization(axis = -1, name = BN_name + '_shortcut')(X_shortcut)

    X = Add()([X, X_shortcut])
    X = relu(X)
    return X

  def ResNet2D(self, save_img=False):

    """ Builds the residual network architecture by calling identity_block_2D and convolutional_block_2D. 
        Puts together structure and depth of model here by stacking identity and convolutional layers.
        More stages/blocks may be added at will.
    """

    # X_input = X_input
    X = ZeroPadding2D((3, 3))(self.X_input)

    # stage 1
    X = Conv2D(filters = 32, 
               kernel_size = (7, 7), 
               strides=(2, 2), 
               kernel_initializer='glorot_uniform', 
               name = 'Conv2D_Stage_1')(X)
    X = BatchNormalization(name = 'BN2D_Stage_1')(X)
    X = relu(X)
    X = MaxPooling2D(pool_size = (3, 3), strides = (2, 2))(X)

    # stage 2
    X = self.convolutional_block_2D(X = X, filters=[32, 32, 126], kernel_size=3, 
                                    stride=1, stage = 2, block = 'A')
    X = self.identity_block_2D(X = X, filters = [32, 32, 126], kernel_size = 3, 
                               stage = 2, block = 'B')
    X = self.identity_block_2D(X = X, filters = [32, 32, 126], kernel_size = 3, 
                               stage = 2, block = 'C')

    # stage 3
    X = self.convolutional_block_2D(X = X, filters=[64, 64, 256], kernel_size=3, 
                                    stride=2, stage = 3, block = 'A')
    X = self.identity_block_2D(X = X, filters = [64, 64, 256], kernel_size = 3, 
                               stage = 3, block = 'B')
    X = self.identity_block_2D(X = X, filters = [64, 64, 256], kernel_size = 3, 
                               stage = 3, block = 'C')
    X = self.identity_block_2D(X = X, filters = [64, 64, 256], kernel_size = 3, 
                               stage = 3, block = 'D')

    # stage 4
    X = self.convolutional_block_2D(X = X, filters=[128, 128, 512], kernel_size=3, 
                                    stride=2, stage = 4, block = 'A')
    X = self.identity_block_2D(X = X, filters =[128, 128, 512], kernel_size = 3, 
                               stage = 4, block = 'B')
    X = self.identity_block_2D(X = X, filters =[128, 128, 512], kernel_size = 3, 
                               stage = 4, block = 'C')
    X = self.identity_block_2D(X = X, filters =[128, 128, 512], kernel_size = 3, 
                               stage = 4, block = 'D')
    X = self.identity_block_2D(X = X, filters =[128, 128, 512], kernel_size = 3, 
                               stage = 4, block = 'E')
    X = self.identity_block_2D(X = X, filters =[128, 128, 512], kernel_size = 3, 
                               stage = 4, block = 'F')

    if self.trim_end: 
      # return feature map and do NOT perform sigmoid activation.
      return X
    else:
      # get output layers and perform sigmoid activation.
      X = Flatten()(X)
      X = Dense(units = self.num_classes, activation = 'sigmoid', name = 'final_dense')(X)
      
      # create model
      model = Model(inputs = self.X_input, outputs = X, name='myResNet')
      if save_img:
        keras.utils.plot_model(model, 'ResNet2D.png')
      return model


if __name__ == '__main__':
  
  # set data path and files 
  data_dir = '/Users/lukeprice/github/multi-modal/datafiles/Y8M_segmented/'
  trainFiles = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if '.tfrecord' in file]

  # build dataset
  tran_data_builder = tf_Data_Builder()
  data_builder.create_dataset(
    tf_datafiles = trainFiles, 
    batch_size = 32)
  data_builder.fit_multi_hot_encoder(
    class_labels = np.array([[170],[1454],[709],[1057],[1308]]))
  
  resnet1D = ResNet(input_shape = (128,1), num_classes = 5)
  resnet1D.ResNet1D()

  resnet2D = ResNet(input_shape = (32,32,1), num_classes = 5)
  resnet2D.ResNet2D()


