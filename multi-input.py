import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras import layers, Input, Model
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import concatenate
from ResNet import ResNet
from segmented_data_builder import tf_Data_Builder
from tqdm import tqdm
import os
import numpy as np
import sys
from sklearn.metrics import multilabel_confusion_matrix
import copy

class Multi_Modal():

  def __init__(self, data_builder_object):
    self.data_builder = data_builder
    self.batch_size = self.data_builder.batch_size

  def compile_multi_modal_network(self, model_summary = True, save_img = False):
    # define input shapes
    image_inputs = Input(
      shape=(32,32,1), 
      name = 'image_Inputs')
    audio_inputs = Input(
      shape=(128,1), 
      name = 'audio_Inputs')
    
    # create 2D ResNet for image data 
    resnet2D = ResNet(trim_front = True, trim_end = True, X_input = image_inputs)
    x_image = resnet2D.ResNet2D()

    # create 1D ResNet for audio data 
    resnet1D = ResNet(trim_front = True, trim_end = True, X_input = audio_inputs)
    x_audio = resnet1D.ResNet1D()

    # get image and audio to contain equal number of units (50/50)
    x_image = Flatten()(x_image)
    x_image = BatchNormalization()(x_image)
    x_image = Dense(units = 1000, activation = 'relu', name = 'dense_image')(x_image)
    x_image = Dropout(0.2)(x_image)
    x_image = BatchNormalization()(x_image)
    x_audio = Flatten()(x_audio)
    x_audio = Dense(units = 1000, activation = 'relu', name = 'dense_audio')(x_audio)
    x_audio = Dropout(0.2)(x_audio)
    x_audio = BatchNormalization()(x_audio)

    # # concatenate inputs 
    x = concatenate([x_image, x_audio])
    x = Dense(units = x.shape[1], activation = 'relu', name = 'Merged_dense_1')(x)
    x = Dense(units = 1000, activation = 'relu', name = 'Merged_dense_2')(x)
    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)

    # # send merged features into 3rd ResNet
    # x = tf.expand_dims(x, -1)
    # merged_resnet = ResNet(trim_front = True, trim_end = True, X_input = x)
    # x = merged_resnet.ResNet1D()
    # print(x)
    # input()
    x = Dense(100, activation='relu', name='Merged_Dense_3')(x)
    
    output_layer = Dense(units = 5, activation='sigmoid', 
      name = 'output_Layer')
    outputs = output_layer(x)

    self.model = Model(
      inputs=[image_inputs, audio_inputs], 
      outputs=outputs)

    if model_summary:
      self.model.summary()
    if save_img:
      keras.utils.plot_model(self.model, 'multi_model.png')


  def train_model(self, epochs, learning_rate = 0.001, predict_after_epoch = False):
    # model = self.compile_multi_modal_network(model_summary = False, save_img = False)
    print('\nTraining Model')
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    loss_function = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    loss_history = []
    for epoch in range(epochs):
      epoch_loss = []
      with tqdm(total = int(self.data_builder.train_size//self.batch_size)) as pbar: 
        # var_list_fn = lambda: self.model.trainable_weights
        for batch, (images, audio, labels) in enumerate(self.data_builder.train_dataset):
          labels = tf.sparse.to_dense(labels)
          multi_hotted_labels = self.data_builder.multi_hot_classes(labels)
          with tf.GradientTape() as tape:
            img = tf.cast(images,tf.float32)
            aud = tf.cast(audio,tf.float32)
            logits = self.model([img, aud])
            loss = loss_function(y_true = multi_hotted_labels, y_pred = logits)
            epoch_loss.append(loss)
            loss_history.append(loss)

          before_weights = copy.deepcopy(self.model.trainable_weights)
          gradients = tape.gradient(target = loss, sources = self.model.trainable_weights)
          optimizer.apply_gradients(zip(gradients, self.model.trainable_weights))
          after_weights = self.model.trainable_weights
      
          # Unit-Test: check to make sure weights were updated
          no_update = False
          for b, a in zip(before_weights, after_weights):
            if all(np.array(tf.equal(a, b)).flatten()):
              print('No weight update at - epoch {} batch {} -'.format(epoch+1,batch))
              no_update = True
              break

            # assert any(np.array(tf.not_equal(a, b)).flatten())
          if batch == 0 or batch % 100 == 0:
            print('Batch {} loss: {}'.format(batch, float(loss)))
          pbar.update(1)

      avg_loss_on_epoch = np.mean(epoch_loss)
      print('Epoch {} avg. training loss = {}'.format(epoch+1,float(avg_loss_on_epoch)))
      print('Epoch {} final batch loss = {}'.format(epoch+1,float(loss)))

      if predict_after_epoch:
        self.predict_model()
        print(np.sum(self.predictions, axis = 0))
        self.get_model_metrics(
          true_labels = self.true_labels, 
          predicted_labels = self.predictions, 
          rates=False)


  def predict_model(self):
    print('\nPredicting Model')
    self.predictions = []
    self.true_labels = []
    with tqdm(total = int(self.data_builder.test_size//self.batch_size)) as pbar: 
      for batch, (images, audio, labels) in enumerate(self.data_builder.test_dataset):
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


  def get_model_metrics(self, true_labels, predicted_labels, rates = True):
    conf_mat = multilabel_confusion_matrix(
      true_labels, 
      predicted_labels)#.ravel()

    conf_mat_sum = np.zeros((2,2))
    for mat in conf_mat:
      conf_mat_sum += mat
    tn, fp, fn, tp = conf_mat_sum.flatten()

    print('\ntp : ' + str(tp))
    print('fp : ' + str(fp))
    print('tn : ' + str(tn))
    print('fn : ' + str(fn))

    if rates:
      tpr = tp / (tp + fn)
      fpr = fp / (fp + tn)
      p = tp / (tp + fp)
      r = tp / (tp + fn)
      precision = tp / (tp + fp)
      recall = tp / (tp + fn)
      f1_score = 2 * ((precision * recall) / (precision + recall))

      print('\nPrecision: {} {}Recall: {}'.format(round(p,3), '\n', round(r,3)))
      print('F1 Score: {}'.format(round(f1_score,3)))


  def get_train_labels(self):
    """ mainly used to analyze class imbalance (np.sum(self.train_labels, 0)) """
    self.train_labels = []
    for batch, (image, audio, labels) in enumerate(self.data_builder.train_dataset):
      labels = tf.sparse.to_dense(labels)
      self.train_labels.extend(np.array(self.data_builder.multi_hot_classes(labels)))
    self.train_labels = np.array(self.train_labels)


  def train_on_N_examples(self, N, epochs, learning_rate=0.001):
    """ N = num_examples; max N is batch_size """
    print('Training on {} examples'.format(N))
    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
    loss_function = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    loss_history = []
    X = self.data_builder.train_dataset
    for batch, (images, audio, labels) in enumerate(self.data_builder.train_dataset):
      img = images[0:N]
      img = tf.reshape(img, [N,32,32,1])
      img = tf.cast(img,tf.float32)
      aud = audio[0:N]
      aud = tf.reshape(aud, [N,128,1])
      aud = tf.cast(aud,tf.float32)
      labels = tf.sparse.to_dense(labels)
      multi_hotted_labels = self.data_builder.multi_hot_classes(labels)
      label = multi_hotted_labels[:N]
      label = tf.reshape(label, [N,5])
      break

    print('True Label: {}\n'.format(label))
    for epoch in range(epochs):
      with tf.GradientTape() as tape:
        logits = self.model([img, aud])

        acc = tf.equal(tf.cast(label, tf.float32), tf.round(logits))
        acc = tf.cast(acc, tf.float32)
        acc = tf.reduce_mean(acc)
        loss = loss_function(y_true = label, y_pred = logits)
        loss_history.append(loss)
      gradients = tape.gradient(target = loss, sources = self.model.trainable_weights)
      optimizer.apply_gradients(zip(gradients, self.model.trainable_weights))
      print('Loss {}'.format(float(loss)))
      print('Accuracy {}\n'.format(float(acc)))
      self.get_model_metrics(label, tf.round(logits), rates = False)
      print('----------------------------------------')
      input()


if __name__ == '__main__':
  
  # set data path and files 
  train_dir = '/Users/lukeprice/github/multi-modal/datafiles/Y8M_segmented/train'
  test_dir = '/Users/lukeprice/github/multi-modal/datafiles/Y8M_segmented/test'
  trainFiles = [os.path.join(train_dir, file) for file in os.listdir(train_dir) if '.tfrecord' in file]
  testFiles = [os.path.join(test_dir, file) for file in os.listdir(test_dir) if '.tfrecord' in file]

  # build dataset
  data_builder = tf_Data_Builder()
  data_builder.fit_multi_hot_encoder(
    class_labels = np.array([[170],[1454],[709],[1057],[1308]]))
  data_builder.create_train_test_dataset(
    train_tf_datafiles = trainFiles, 
    test_tf_datafiles = testFiles, 
    batch_size = 32)  

  # build, train, test, evaluate model 
  model = Multi_Modal(data_builder)
  model.compile_multi_modal_network(False, True)
  model.train_model(
    epochs = 100, 
    learning_rate = 0.00001, 
    predict_after_epoch = True)
  model.get_model_metrics(
    true_labels = self.true_labels, 
    predicted_labels = self.predictions)

  """
  Label Imbalance 
  Train-True
  print(np.sum(model_object.train_labels, axis = 0))
  [15333.  3578.  3012.  2105.  2219.] 
  Test-True
  print(np.sum(model_object.true_labels, axis = 0))
  [2110.  494.  399.  275.  270.]
  Test-Predicted
  print(np.sum(model_object.predictions, axis = 0))
  ...
  """





