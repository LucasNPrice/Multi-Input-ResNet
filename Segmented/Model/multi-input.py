import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras import layers
from tensorflow.keras.layers import MaxPooling2D, BatchNormalization
from ResNet import ResNet
from segmented_data_builder import tf_Data_Builder
from tqdm import tqdm
import os
import numpy as np
import sys
from sklearn.metrics import multilabel_confusion_matrix

class Multi_Modal():
  def __init__(self, data_builder_object):
    self.data_builder = data_builder

  def convAudioNet(self, inputs):
    conv1D_layer = layers.Conv1D(
      filters = 32,
      kernel_size = 3,
      activation = 'relu')
    dense_layer = layers.Dense(
      units = 32, 
      activation = 'relu',
      name = 'dense_Audio')
    x = conv1D_layer(inputs)
    x = BatchNormalization()(x)
    # x = dense_layer(inputs)
    # x = BatchNormalization()(x)
    # x = layers.Flatten()(x)
    return x

  def compile_multi_modal_network(self, model_summary = True, save_img = False):
    # define input shapes
    image_inputs = keras.Input(
      shape=(32,32,1), 
      name = 'image_Inputs')

    audio_inputs = keras.Input(
      shape=(128,1), 
      name = 'audio_Inputs')
    
    resnet_model = ResNet(trim_front = True, trim_end = True, X_input = image_inputs)
    x_image = resnet_model.resnet()
    # send audio to 1D conv layers
    x_audio = self.convAudioNet(audio_inputs)

    # flatten for dense layer
    x_image = layers.Flatten()(x_image)
    x_audio = layers.Flatten()(x_audio)

    # get image and audio to equal number of units 50/50
    x_image = layers.Dense(units = 500, activation = 'relu', name = 'dense_image')(x_image)
    x_audio = layers.Dense(units = 500, activation = 'relu', name = 'dense_audio')(x_audio)

    x = layers.concatenate([x_image, x_audio])
    x = layers.Dense(100, activation='relu', name='First_Dense')(x)
    
    output_layer = layers.Dense(
      units = 5, 
      activation='sigmoid', 
      name = 'output_Layer')
    outputs = output_layer(x)

    self.model = keras.Model(
      inputs=[image_inputs, audio_inputs], 
      outputs=outputs)
    if model_summary:
      self.model.summary()
    if save_img:
      keras.utils.plot_model(self.model, 'multi_model.png')
    # return model

  def train_model(self, epochs):
    # model = self.multi_modal_network(True)
    print('\nTraining Model')
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
            logits = self.model([img, aud])
            loss = loss_function(y_true = multi_hotted_labels, y_pred = logits)
            epoch_loss.append(loss)
            loss_history.append(loss)
          gradients = tape.gradient(target = loss, sources = self.model.trainable_weights)
          optimizer.apply_gradients(zip(gradients, self.model.trainable_weights))
          pbar.update(1)
      avg_loss_on_epoch = np.mean(epoch_loss)
      print('Epoch {} avg. loss = {}'.format(epoch+1,float(avg_loss_on_epoch)))
      print('Epoch {} final batch loss = {}'.format(epoch+1,float(loss)))

  def predict_model(self):
    print('\nPredicting Model')
    self.predictions = []
    self.true_labels = []
    with tqdm(total = 29545 // 32) as pbar: 
      for batch, (images, audio, labels) in enumerate(self.data_builder.dataset):
        img = tf.cast(images,tf.float32)
        aud = tf.cast(audio,tf.float32)
        logit_predictions = self.model([img, aud])
        logit_predictions = tf.round(logit_predictions)
        self.predictions.extend(np.array(logit_predictions))
        labels = tf.sparse.to_dense(labels)
        self.true_labels.extend(np.array(self.data_builder.multi_hot_classes(labels)))
        pbar.update(1)
    self.predictions = np.array(self.predictions)
    self.true_labels = np.array(self.true_labels)

  def get_model_metrics(self):
    conf_mat = multilabel_confusion_matrix(
      self.true_labels, 
      self.predictions)#.ravel()

    conf_mat_sum = np.zeros((2,2))
    for mat in conf_mat:
      conf_mat_sum += mat

    tn, fp, fn, tp = conf_mat_sum.flatten()
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * ((precision * recall) / (precision + recall))

    print('\ntp : ' + str(tp))
    print('fp : ' + str(fp))
    print('tn : ' + str(tn))
    print('fn : ' + str(fn))
    print('\nPrecision: {} {}Recall: {}'.format(round(p,3), '\n', round(r,3)))
    print('F1 Score: {}'.format(round(f1_score,3)))


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

  # create model object plus train, predict, and evaluate
  # get a ResNet 
  # resnet = ResNet(trim_front = True, trim_end = True)
  # sys.exit()
  model_object = Multi_Modal(data_builder)
  model_object.compile_multi_modal_network(True, True)
  model_object.train_model(1)
  model_object.predict_model()
  model_object.get_model_metrics()


