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
from tqdm import trange
import os
import numpy as np
import sys
import copy
import pickle

from ResNet import ResNet
from segmented_data_builder import tfDataBuilder
from focal_loss import FocalLoss
from metrics import Metrics


class MultiModal():

  def __init__(self, data_builder_object):

    """ Creates an object of class MultiModal.

    Must be called with tfDataBuilder object containng a tensorflow Dataset object. 

    Args:
      data_builder_object: Object of class tfDataBuilder. 
    """

    self.data_builder = data_builder
    self.batch_size = self.data_builder.batch_size

  def compile_json_model(self, json_model, **kwargs):

    """ Compile a model from a .json file.

    Args:
      json_model: Model saved to a .json file.
      weights: (Optional) pretrained weights compatible with the .json model.
    """

    json_file = open(json_model, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    self.model = model_from_json(loaded_model_json)
    if 'weights' in kwargs.keys():
      weights = kwargs['weights']
      self.model.load_weights(weights)  

  def compile_multi_modal_network(self, 
                                  model_summary=True, 
                                  save_img=False, 
                                  img_name='multi_model.png',
                                  save_json=False, 
                                  json_file_name='multi_modal_model.json'):

    """ Builds and compiles a multi-input residual network of class ResNet. 

    Args:
      model_summary: True to display model summary; boolean. 
      save_img: True to save image of model architecture to .png file; boolean.
      save_json: True to save model architecture to .json file format; boolean.
      json_file_name: name of json file to write to. 
    """ 

    image_inputs = Input(shape=(32,32,1), 
                         name = 'image_Inputs')
    audio_inputs = Input(shape=(128,1), 
                         name = 'audio_Inputs')
    
    resnet2D = ResNet(trim_front=True, trim_end=True, X_input=image_inputs)
    x_image = resnet2D.ResNet2D()
    resnet1D = ResNet(trim_front=True, trim_end=True, X_input=audio_inputs)
    x_audio = resnet1D.ResNet1D()

    x_image = Flatten()(x_image)
    x_image = BatchNormalization()(x_image)
    x_image = Dense(units=1000, activation='relu', name='dense_image')(x_image)
    x_image = Dropout(0.2)(x_image)
    x_image = BatchNormalization()(x_image)
    x_audio = Flatten()(x_audio)
    x_audio = BatchNormalization()(x_audio)
    x_audio = Dense(units=1000, activation='relu', name='dense_audio')(x_audio)
    x_audio = Dropout(0.2)(x_audio)
    x_audio = BatchNormalization()(x_audio)

    x = concatenate([x_image, x_audio])
    x = Dense(units=x.shape[1], activation='relu', name='Merged_dense_1')(x)
    x = Dense(units=1000, activation='relu', name='Merged_dense_2')(x)
    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)
    x = Dense(100, activation='relu', name='Merged_Dense_3')(x)

    output_layer = Dense(units=5, activation='sigmoid', name='output_Layer')
    outputs = output_layer(x)
    self.model = Model(inputs=[image_inputs, audio_inputs], outputs=outputs)

    if model_summary:
      self.model.summary()
    if save_img:
      keras.utils.plot_model(self.model, img_name)
    if save_json:
      model_json = self.model.to_json()
      with open(json_file_name, 'w') as json_file:
        json_file.write(model_json)
        # json_file.close()

  def train_model(self, epochs, loss_function, 
                  learning_rate=0.001, 
                  metrics = ['loss'],
                  predict_after_epoch=False, 
                  save_weights=False,
                  weights_file_name=None, 
                  save_metrics=False,
                  metrics_file_name=None,
                  assert_weight_update=False,
                  **kwargs):

    """ Trains a multi-input model of class ResNet. 

    Args:
      epochs: Number of epochs to train model. 
      loss_function: Loss function to train model. 
      learning_rate: Learning rate of loss function. 
      metrics: list of metrics to record and print during training. 
      predict_after_epoch: True to evaluate model (on new data) after each epoch. 
      save_weights: True to save model weights to .h5 file format. Currently only saves best weights as determined by F1 on validation data.
        If True, weights_file_name must be defined.  
      weights_file_name: File name to save weights to. Must be .h5 file. 
      save_metrics: True to save training metrics as training progresses. The metrics saved are set in parameter 'metrics'.
        If True, metrics_file_name must be defined. 
      metrics_file_name: File name to save metrics to. Must be .pickle or .json.
      assert_weight_update: Assert that weights are updated after each batch. If true, and 
        weights are equal to the weights prior to update, assertion is triggered and model stops training. 
    """

    if save_weights:
      assert weights_file_name is not None, 'weights_file_name must be defined if save_weights=True'
      assert predict_after_epoch, 'predict_after_epoch must be True if save_weights=True'
    if save_metrics:
      assert metrics_file_name is not None, 'metrics_file_name must be defined if save_metrics=True'
      if not os.path.exists(metrics_file_name):
        logged_metrics = {}
        Epochs = {}
        epoch_start = 0
        best_F1 = 0
      else:
        with open(metrics_file_name, 'rb') as file:
          logged_metrics = pickle.load(file)
        Epochs = logged_metrics['Epochs']
        best_F1 = 0
        for epoch in Epochs:
          epoch_F1 = Epochs[epoch]['metrics']['F1']
          if epoch_F1 > best_F1:
            best_F1 = epoch_F1
        epoch_start = sorted(logged_metrics['Epochs'])[-1] + 1
        epochs += epoch_start
    self.metrics = metrics
    print(logged_metrics)
    input()

    print('\nTraining Model')
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    for epoch in range(epoch_start,epochs):
      num_batches = int(self.data_builder.train_size//self.batch_size)
      with tqdm(total=num_batches) as pbar:
        for batch, (images, audio, labels) in enumerate(self.data_builder.train_dataset):
          labels = tf.sparse.to_dense(labels)
          multi_hotted_labels = self.data_builder.multi_hot_classes(labels)

          with tf.GradientTape() as tape:
            img = tf.cast(images,tf.float32)
            aud = tf.cast(audio,tf.float32)
            logits = self.model([img, aud])
            loss = loss_function(y_true=multi_hotted_labels, y_pred=logits)

          if assert_weight_update:
            before_weights = copy.deepcopy(self.model.trainable_weights)

          gradients = tape.gradient(target=loss, 
                                    sources=self.model.trainable_weights)
          optimizer.apply_gradients(zip(gradients, self.model.trainable_weights))

          # Unit-Test: check to make sure weights were updated
          if assert_weight_update:
            after_weights = self.model.trainable_weights
            for b, a in zip(before_weights, after_weights):
              assert any(np.array(tf.not_equal(a, b)).flatten())

          # print latest training loss 
          if batch == 0 or batch % 100 == 0:
            print('Batch {} loss: {}'.format(batch, float(loss)))
          pbar.update(1)

      # if predict_after_epoch, get predictions and display metrics 
      if predict_after_epoch:
        metrics = self.__get_metrics(loss_function)
        print('Epoch {} Validation Loss: {}'.format(epoch+1, metrics['metrics']['loss']))
        print('Epoch {} Validation F1: {}\n'.format(epoch+1, metrics['metrics']['F1']))

        # if save_weights, save weights if current F1 is best F1
        if save_weights:
          if metrics['metrics']['F1'] > best_F1:
            self.model.save_weights(weights_file_name)
          best_F1 = metrics['metrics']['F1']

        # if save metrics, save metrics to .pickel (.json)
        if save_metrics:
          Epochs[epoch] = metrics
          Parameters = {
            'optimizer': optimizer,
            'lr': learning_rate,
            'lossFn': loss_function
          }
          logged_metrics['Epochs'] = Epochs
          with open(metrics_file_name, 'wb') as file:
            print(metrics_file_name)
            pickle.dump(logged_metrics, file, protocol=pickle.HIGHEST_PROTOCOL)

  def __get_metrics(self, loss_function):

    """ Gets validation metrics during training and creates metric dictionary for saving to .pickel/.json format. """

    self.predict_model()
    Epoch = {}
    metrics = {}
    evaluated_metrics = Metrics(self.true_labels, self.predictions)

    if 'loss' in self.metrics:
      validation_loss = loss_function(y_true=self.true_labels, 
                                      y_pred=self.predictions)
      metrics['loss'] = float(validation_loss)
    if 'F1' in self.metrics: 
      F1 = evaluated_metrics.get_F1(return_metric=True)
      metrics['F1'] = F1

    Epoch['metrics'] = metrics

    return Epoch

  def predict_model(self):

    """ Makes predictions using a trained model of class ResNet on new data. """

    print('\nPredicting Model')
    self.predictions = []
    self.true_labels = []
    num_batches = int(tf.round(self.data_builder.test_size/self.batch_size))

    with tqdm(total = num_batches) as pbar: 
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

  def get_train_labels(self):

    """ Returns the true labels/classes of the Dataset. 
        Mainly used to analyze class imbalance (np.sum(self.train_labels, 0)) or 
          for weighted loss functions.  
    """

    self.train_labels = []
    for batch, (image, audio, labels) in enumerate(self.data_builder.train_dataset):
      labels = tf.sparse.to_dense(labels)
      self.train_labels.extend(np.array(self.data_builder.multi_hot_classes(labels)))
    self.train_labels = np.array(self.train_labels)

  def get_label_ratios(self):

    """ Get the ratio of each class in the Dataset. 
        Mainly used for analytics and for weighted loss functions.
    """

    self.get_train_labels()
    self.label_ratios = np.sum(self.train_labels,axis=0)/len(self.train_labels)


if __name__ == '__main__':

  def filepaths(mode):
    dir_ = '/Users/lukeprice/github/multi-modal'

    if mode == 'multi':
      json = dir_ + '/saved_models/multi_modal_model.json'
      weights = dir_ + '/saved_models/multi_modal_weights.h5'
      metrics = dir_ + '/metric_files/multi_modal_metrics.pickle'
    elif mode == 'image':
      json = dir_ + '/saved_models/image_only_model.json'
      weights = dir_ + '/saved_models/image_only_weights.h5'
      metrics = dir_ + '/metric_files/image_only_metrics.pickle'
    elif mode == 'audio':
      json = dir_ + '/saved_models/audio_only_model.json'
      weights = dir_ + '/saved_models/audio_only_weights.h5'
      metrics = dir_ + '/metric_files/audio_only_metrics.pickle'
    else:
      raise NameError('mode must be one of {}'.format('\'multi\'', '\'image\', \'audio\''))

    return json, weights, metrics

  def train_new_model(data_builder, json_file_name, weights_file_name, metrics_file_name):
    """ build, compile, train, test, and evaluate new model """
    model = MultiModal(data_builder)
    model.compile_multi_modal_network(model_summary=False, 
                                      save_img=False, 
                                      save_json=True,
                                      json_file_name=json_file_name)
    model.get_label_ratios()
    focal_loss = FocalLoss(alpha=model.label_ratios, class_proportions=True)
    model.train_model(epochs=10,  
                      loss_function=focal_loss,
                      learning_rate=0.00001, 
                      metrics=['loss','F1'],
                      predict_after_epoch=True,
                      save_weights=True,
                      weights_file_name=weights_file_name,
                      save_metrics=True,
                      metrics_file_name=metrics_file_name,
                      assert_weight_update=True
                      )
    model.predict_model()

  def transfer_model_train(data_builder, json_file_name, weights_file_name, metrics_file_name):
    """ Run with pretrained model """
    transfer_model = MultiModal(data_builder)
    transfer_model.compile_json_model(json_model=json_file_name,
                                      weights=weights_file_name)
    # transfer_model.compile_multi_modal_network(model_summary=False, save_img=True, save_json=True)
    transfer_model.get_label_ratios()
    focal_loss = FocalLoss(alpha=transfer_model.label_ratios, class_proportions=True)
    transfer_model.train_model(epochs=10,  
                              loss_function=focal_loss,
                              learning_rate=0.00001, 
                              metrics=['loss','F1'],
                              predict_after_epoch=True,
                              save_weights=True,
                              save_metrics=True,
                              assert_weight_update=True,
                              weights_file_name=weights_file_name,
                              metrics_file_name=metrics_file_name
                              )

      # set data path and files 
  train_dir = '/Users/lukeprice/github/multi-modal/datafiles/Y8M_segmented/train'
  test_dir = '/Users/lukeprice/github/multi-modal/datafiles/Y8M_segmented/test'
  trainFiles = [os.path.join(train_dir, file) for file in os.listdir(train_dir) if '.tfrecord' in file]
  testFiles = [os.path.join(test_dir, file) for file in os.listdir(test_dir) if '.tfrecord' in file]

  # build dataset (currently only classifying 5 labels)
  data_builder = tfDataBuilder()
  data_builder.fit_multi_hot_encoder(class_labels=np.array([[170],[1454],[709],[1057],[1308]]))
  data_builder.create_train_test_dataset(train_tf_datafiles=trainFiles, 
                                         test_tf_datafiles=testFiles, 
                                         batch_size=32)  

  json_file, weights_file, metrics_file = filepaths('multi')
  train_new_model(data_builder=data_builder,
                  json_file_name=json_file,
                  weights_file_name=weights_file,
                  metrics_file_name=metrics_file)
  transfer_model_train(data_builder=data_builder,
                       json_file_name=json_file,
                       weights_file_name=weights_file,
                       metrics_file_name=metrics_file)
  """
  -----------------------------------------------------
  Label Imbalance 
  -----------------------------------------------------
  Train-True
  print(np.sum(model_object.train_labels, axis = 0))
  [15333.  3578.  3012.  2105.  2219.] 
  -----------------------------------------------------
  Test-True
  print(np.sum(model_object.true_labels, axis = 0))
  [2110.  494.  399.  275.  270.]
  -----------------------------------------------------
  """





