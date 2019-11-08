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

  def compile_multi_modal_network(self, model_summary=True, 
                                  save_img=False, 
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
    x_audio = Dense(units=1000, activation='relu', name='dense_audio')(x_audio)
    x_audio = Dropout(0.2)(x_audio)
    x_audio = BatchNormalization()(x_audio)

    x = concatenate([x_image, x_audio])
    x = Dense(units=x.shape[1], activation='relu', name='Merged_dense_1')(x)
    x = Dense(units=1000, activation='relu', name='Merged_dense_2')(x)
    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)

    # # send merged features into 3rd ResNet
    # x = tf.expand_dims(x, -1)
    # merged_resnet = ResNet(trim_front = True, trim_end = True, X_input = x)
    # x = merged_resnet.ResNet1D()
    # print(x)
    # input()
    x = Dense(100, activation='relu', name='Merged_Dense_3')(x)
    output_layer = Dense(units=5, activation='sigmoid', name='output_Layer')
    outputs = output_layer(x)
    self.model = Model(inputs=[image_inputs, audio_inputs], outputs=outputs)

    if model_summary:
      self.model.summary()
    if save_img:
      keras.utils.plot_model(self.model, 'multi_model.png')
    if save_json:
      model_json = self.model.to_json()
      with open(json_file_name, 'w') as json_file:
        json_file.write(model_json)
        # json_file.close()

  def train_model(self, epochs, loss_function, 
                  learning_rate=0.001, 
                  predict_after_epoch=False, 
                  save_weights=False, 
                  weight_file_name='multi_modal_model.h5', 
                  assert_weight_update=False):

    """ Trains a multi-input model of class ResNet. 

    Args:
      epochs: Number of epochs to train model. 
      loss_function: Loss function to train model. 
      learning_rate: Learning rate of loss function. 
      predict_after_epoch: True to evaluate model (on new data) after each epoch; boolean. 
      save_weights: True to save model weights to .h5 file format; currently only saves best weights as determined by F1 on validation data. 
      weight_file_name: File name to save weights as. 
      assert_weight_update: Assert that weights are updated after each batch. If true, and 
        weights are equal to the weights prior to update, assertion is triggered and model stops training. 
    """

    print('\nTraining Model')
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    if predict_after_epoch:
      self.F1_history = []

    for epoch in range(epochs):
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

          if batch == 0 or batch % 100 == 0:
            print('Batch {} loss: {}'.format(batch, float(loss)))
          pbar.update(1)
      print('Epoch {} final batch loss = {}'.format(epoch+1,float(loss)))

      if predict_after_epoch:
        self.predict_model()
        metrics = Metrics(self.true_labels, self.predictions)
        F1 = metrics.get_F1(return_metric=True)
        self.F1_history.append(F1)
        validation_loss = loss_function(y_true=self.true_labels, 
                                        y_pred=self.predictions)
        print('Epoch {} Validation F1: {}'.format(epoch+1,float(F1)))
        print('Epoch {} Validation Loss: {}\n'.format(epoch+1, float(validation_loss)))

        if save_weights:
          if epoch == 0:
            pass
            # self.model.save_weights(weight_file_name)
          elif all([self.F1_history[-1] > f1 for f1 in self.F1_history[:-2]]):
            print('Updated Model at Epoch {}'.format(epoch+1))
            print('Current F1: {}'.format(self.F1_history[-1]))
            print('F1 History: {}'.format(self.F1_history[:-2]))
            self.model.save_weights(weight_file_name)
          pass

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
        Mainly used for analytics and for weighted loss functions 
    """

    self.get_train_labels()
    self.label_ratios = np.sum(self.train_labels,axis=0)/len(self.train_labels)


if __name__ == '__main__':

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

  # """ build, compile, train, test, and evaluate new model """
  # model = MultiModal(data_builder)
  # model.compile_multi_modal_network(model_summary=False, save_img=True, save_json=True)
  # model.get_label_ratios()
  # focal_loss = FocalLoss(alpha=model.label_ratios, class_proportions=True)
  # model.train_model(epochs=100,  
  #                   loss_function=focal_loss,
  #                   learning_rate=0.00001, 
  #                   predict_after_epoch=True,
  #                   save_weights=True,
  #                   assert_weight_update=True)
  # model.predict_model()
  # metrics = Metrics(self.true_labels, self.predictions)

  # sys.exit()
  """ Run with pretrained model """
  transfer_model = MultiModal(data_builder)
  transfer_model.compile_json_model(json_model='/Users/lukeprice/github/multi-modal/multi_modal_model.json', 
                                    weights='/Users/lukeprice/github/multi-modal/multi_modal_model.h5')
  # transfer_model.compile_multi_modal_network(model_summary=False, save_img=True, save_json=True)
  transfer_model.get_label_ratios()
  focal_loss = FocalLoss(alpha=transfer_model.label_ratios, class_proportions=True)
  transfer_model.train_model(epochs=100,  
                            loss_function=focal_loss,
                            learning_rate=0.00001, 
                            predict_after_epoch=True,
                            save_weights=True,
                            assert_weight_update=True)


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





