import tensorflow as tf
import os
from segmented_data_builder import tf_Data_Builder
import numpy as np
from tensorflow import optimizers
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Activation, Flatten, Dense, MaxPooling2D, Dropout, BatchNormalization
from tensorflow import optimizers
""" 29545 records sharded across 41 .tfrecord files! """

class Model():
  def __init__(self, data_builder_object):
    self.data_builder = data_builder_object

  def build_model(self):
    self.model = Sequential()

    self.model.add(Conv2D(input_shape = (32, 32, 1),
      filters = 64, 
      kernel_size = 3,
      activation = 'elu'))
    # self.model.add(Conv2D(filters = 64, 
    #   kernel_size = 3,
    #   activation = 'elu'))
    # self.model.add(BatchNormalization())
    self.model.add(MaxPooling2D(pool_size=(2, 2)))
    self.model.add(Dropout(0.2))

    # self.model.add(Conv2D(filters = 128, 
    #   kernel_size = 3,
    #   activation = 'elu'))
    # self.model.add(Conv2D(filters = 128, 
    #   kernel_size = 3,
    #   activation = 'elu'))
    # self.model.add(BatchNormalization())
    # self.model.add(MaxPooling2D(pool_size=(2, 2)))
    # self.model.add(Dropout(0.2))

    # self.model.add(Conv2D(filters = 256, 
    #   kernel_size = 3,
    #   activation = 'elu'))
    # self.model.add(Conv2D(filters = 256, 
    #   kernel_size = 3,
    #   activation = 'elu'))
    # self.model.add(BatchNormalization())
    # # self.model.add(MaxPooling2D(pool_size=(2, 2)))
    # self.model.add(Dropout(0.2))

    self.model.add(Flatten())
    # self.model.add(Dense(512, activation = 'elu'))
    # self.model.add(Dropout(0.2))
    # self.model.add(Dense(128, activation = 'elu'))
    self.model.add(Dense(64, activation = 'elu'))
    self.model.add(Dense(5, activation = 'sigmoid'))
    # return model

  def train(self,epochs):
    optimizer = tf.keras.optimizers.Adam(lr=0.0005)
    loss_function = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    loss_history = []
    for epoch in range(epochs):
      epoch_loss = []
      with tqdm(total = 29545 // 32) as pbar: 
        for batch, (images, labels) in enumerate(self.data_builder.dataset):
          labels = tf.sparse.to_dense(labels)
          multi_hotted_labels = self.data_builder.multi_hot_classes(labels)
          with tf.GradientTape() as tape:
            # get logits/outputs by sending images to model 
            logits = self.model(tf.cast(images,tf.float32))
            # send logits to loss function to get loss
            loss = loss_function(y_true = multi_hotted_labels, y_pred = logits)
            epoch_loss.append(loss)
            loss_history.append(loss)
          """ get gradients of weights """
          gradients = tape.gradient(target = loss, sources = self.model.trainable_weights)
          """ update weights with optimizer """
          optimizer.apply_gradients(zip(gradients, self.model.trainable_weights))
          pbar.update(1)
      """ log results during training """
      avg_loss_on_epoch = np.mean(epoch_loss)
      print('Epoch {} avg. loss = {}'.format(epoch+1,float(avg_loss_on_epoch)))
      print('Epoch {} final batch loss = {}'.format(epoch+1,float(loss)))

  def predict(self):
    self.predictions = []
    self.true_labels = []
    for batch, (images, labels) in enumerate(self.data_builder.dataset):
      logit_predictions = self.model(tf.cast(images,tf.float32))
      logit_predictions = tf.round(logit_predictions)
      self.predictions.extend(logit_predictions)
      labels = tf.sparse.to_dense(labels)
      self.true_labels.extend(self.data_builder.multi_hot_classes(labels))


if __name__ == '__main__':
  
  data_dir = '/Users/lukeprice/github/multi-modal/datafiles/Y8M_segmented/'
  trainFiles = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if '.tfrecord' in file]

  data_builder = tf_Data_Builder()
  data_builder.create_dataset(
    tf_datafiles = trainFiles, 
    batch_size = 32)
  data_builder.fit_multi_hot_encoder(
    class_labels = np.array([[170],[1454],[709],[1057],[1308]]))

  NN = Model(data_builder_object = data_builder)
  NN.build_model()
  NN.train(epochs = 1)
  NN.predict()
  sys.exit()




  # epochs = 3
  # optimizer = tf.keras.optimizers.Adam(lr=0.005)
  # loss_function = tf.keras.losses.BinaryCrossentropy(from_logits=True)
  # loss_history = []

  # """ for each epochs """
  # for epoch in range(epochs):
  #   print('Epoch %d' % (epoch+1,))
  #   epoch_loss = []
  #   with tqdm(total = 29545 // 32) as pbar: 
  #     """ for each batch in epoch """
  #     for batch, (images, labels) in enumerate(data_builder.dataset):
  #       labels = tf.sparse.to_dense(labels)
  #       multi_hotted_labels = data_builder.multi_hot_classes(labels)
  #       """ begin gradient tape """
  #       with tf.GradientTape() as tape:
  #         # get logits/outputs by sending images to model 
  #         logits = model(tf.cast(images,tf.float32))
  #         # send logits to loss function to get loss
  #         loss = loss_function(y_true = multi_hotted_labels, y_pred = logits)
  #         epoch_loss.append(loss)
  #         loss_history.append(loss)
  #       """ get gradients of weights """
  #       gradients = tape.gradient(target = loss, sources = model.trainable_weights)
  #       """ update weights with optimizer """
  #       optimizer.apply_gradients(zip(gradients, model.trainable_weights))
  #       pbar.update(1)
  #   """ log results during training """
  #   avg_loss_on_epoch = np.mean(epoch_loss)
  #   print('Epoch {} avg. loss = {}'.format(epoch+1,float(avg_loss_on_epoch)))
  #   print('Epoch {} final batch loss = {}'.format(epoch+1,float(loss)))


  # plt.plot(loss_history)
  # plt.xlabel('Batch #')
  # plt.ylabel('Loss [entropy]')
  # plt.show()

