import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras import layers, Input, Model
from tensorflow.keras.layers import ZeroPadding2D, Conv2D
from tensorflow.keras.layers import MaxPooling2D, BatchNormalization, Flatten, Dense, concatenate
from ResNet import ResNet
from segmented_data_builder import tf_Data_Builder
from tqdm import tqdm
import os
import numpy as np
import sys
from sklearn.metrics import multilabel_confusion_matrix
import copy


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



  image_inputs = Input(
      shape=(32,32,1), 
      name = 'image_Inputs')
  x = Conv2D(filters = 32, kernel_size = (1,1), strides = (1,1), padding = 'valid')(image_inputs)
  x = Flatten()(x)
  output_layer = Dense(units = 5, activation='sigmoid', 
      name = 'output_Layer')
  outputs = output_layer(x)
  model = Model(inputs=image_inputs, outputs=outputs)
  model.summary()
  input()

  optimizer = tf.keras.optimizers.Adam()
  loss_function = tf.keras.losses.BinaryCrossentropy(from_logits=True)
  loss_history = []
  for epoch in range(3):
    epoch_loss = []
    with tqdm(total = int(data_builder.train_size)) as pbar: 
      # var_list_fn = lambda: self.model.trainable_weights
      for batch, (images, audio, labels) in enumerate(data_builder.train_dataset):
        labels = tf.sparse.to_dense(labels)
        multi_hotted_labels = data_builder.multi_hot_classes(labels)
        with tf.GradientTape() as tape:
          img = tf.cast(images,tf.float32)
          # aud = tf.cast(audio,tf.float32)
          # logits = self.model([img, aud])
          logits = model(img)
          loss = loss_function(y_true = multi_hotted_labels, y_pred = logits)
          epoch_loss.append(loss)
          loss_history.append(loss)

        import copy 
        before_weights = copy.deepcopy(model.trainable_weights)
        print('BEFORE WEIGHTS')
        print(before_weights)
        input()
        # optimizer.minimize(loss, var_list=model.trainable_weights)
        # input()
        gradients = tape.gradient(target = loss, sources = model.trainable_weights)
        print('GRADIENTS')
        print(gradients)
        input()
        opt_op = optimizer.apply_gradients(zip(gradients, model.trainable_weights))
        print('OPT_OP')
        print(opt_op)
        input()

        after_weights = model.trainable_weights
        print('AFTER WEIGHTS')
        print(after_weights)
        input()
        for b, a in zip(before_weights, after_weights):
          print('BEFORE WEIGHTS')
          print(b)
          print('AFTER WEIGHTS')
          print(a)
          print(any(np.array(tf.not_equal(a, b)).flatten()))
          input()
        pbar.update(1)

  sys.exit()