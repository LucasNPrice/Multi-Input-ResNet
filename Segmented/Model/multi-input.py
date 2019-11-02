import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras import layers
from tensorflow.keras.layers import MaxPooling2D, BatchNormalization
from segmented_data_builder import tf_Data_Builder
from tqdm import tqdm
import os
import numpy as np
import sys

class Multi_Modal():
  def __init__(self, data_builder_object):
    self.data_builder = data_builder

  def convImageNet(self, inputs):
    conv2D_layer = layers.Conv2D(
      filters = 64, 
      kernel_size = 3,
      activation = 'elu')
    x = conv2D_layer(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    flat_layer = layers.Flatten()
    x = flat_layer(x)
    return x

  def convAudioNet(self, inputs):
    conv1D_layer = layers.Conv1D(
      filters = 32,
      kernel_size = 3,
      activation = 'relu')
    dense_layer = layers.Dense(
      units = 32, 
      activation = 'relu',
      name = 'dense_Audio')
    # x = conv1D_layer(inputs)
    x = dense_layer(inputs)
    x = BatchNormalization()(x)
    flat_layer = layers.Flatten()
    x = flat_layer(x)
    return x

  def multi_modal_network(self, save_img = False):

    image_inputs = keras.Input(
      shape=(32,32,1), 
      name = 'image_Inputs')
    audio_inputs = keras.Input(
      shape=(128,1), 
      name = 'audio_Inputs')
    x_image = self.convImageNet(image_inputs)
    x_audio = self.convAudioNet(audio_inputs)
    x = layers.concatenate([x_image, x_audio])
    x = layers.Dense(100, activation='relu', name='First_Dense')(x)
    
    output_layer = layers.Dense(
      units = 5, 
      activation='sigmoid', 
      name = 'output_Layer')
    outputs = output_layer(x)

    model = keras.Model(
      inputs=[image_inputs, audio_inputs], 
      outputs=outputs)

    model.summary()
    if save_img:
      keras.utils.plot_model(model, 'multi_model.png')
    return model

  def train(self, epochs):
    model = self.multi_modal_network(True)
    optimizer = tf.keras.optimizers.Adam(lr=0.0005)
    loss_function = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    loss_history = []
    for epoch in range(epochs):
      epoch_loss = []
      with tqdm(total = 29545 // 32) as pbar: 
        for batch, (images, audio, labels) in enumerate(self.data_builder.dataset):
          labels = tf.sparse.to_dense(labels)
          multi_hotted_labels = self.data_builder.multi_hot_classes(labels)
          with tf.GradientTape() as tape:
            img = tf.cast(images,tf.float32)
            aud = tf.cast(audio,tf.float32)
            logits = model([img, aud])
            loss = loss_function(y_true = multi_hotted_labels, y_pred = logits)
            epoch_loss.append(loss)
            loss_history.append(loss)
          gradients = tape.gradient(target = loss, sources = model.trainable_weights)
          optimizer.apply_gradients(zip(gradients, model.trainable_weights))
          pbar.update(1)
      avg_loss_on_epoch = np.mean(epoch_loss)
      print('Epoch {} avg. loss = {}'.format(epoch+1,float(avg_loss_on_epoch)))
      print('Epoch {} final batch loss = {}'.format(epoch+1,float(loss)))

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
  model_object = Multi_Modal(data_builder)
  model_object.train(2)
  # run model 
  # model = multi_modal_network()
  # for batch, (images, audio, labels) in enumerate(data_builder.dataset):
  #   img = tf.cast(images,tf.float32)
  #   aud = tf.cast(audio,tf.float32)
  #   logits = model([img, aud])



