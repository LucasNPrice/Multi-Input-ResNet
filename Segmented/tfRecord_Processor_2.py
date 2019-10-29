import tensorflow as tf
from tqdm import tqdm
import os 
import numpy as np

class tf_Record_Processor():
  def __init__(self):
    pass

  def write_segments(self, inDir, outDir):
    inDirFiles = os.listdir(inDir)
    for file in inDirFiles:
      if '.tfrecord' in file:
        outfile = os.path.join(outDir, file)
        raw_tfrecord = os.path.join(inDir, file)
        dataset = tf.data.TFRecordDataset(raw_tfrecord)
        with tf.io.TFRecordWriter(outfile) as tfwriter:
          for raw_example in dataset:
            self.__split_on_frames(raw_example, tfwriter)

  def __split_on_frames(self, raw_example, tfwriter):
    example = tf.train.SequenceExample.FromString(raw_example.numpy())
    image = example.feature_lists.feature_list['rgb']
    audio = example.feature_lists.feature_list['audio']
    n_frames = len(image.feature)
    for i in range (0, n_frames):
      img_frame = image.feature[i].bytes_list.value
      audio_frame = audio.feature[i].bytes_list.value
      # tf.train.Feature(bytes_list=tf.train.BytesList(value=img_frame))
      new_features = tf.train.Features(feature={
        'id': example.context.feature['id'],
        'frame': tf.train.Feature(int64_list=tf.train.Int64List(value=[i])),
        'labels': example.context.feature['labels'],
        'rgb': tf.train.Feature(bytes_list=tf.train.BytesList(value=img_frame)),
        'audio': tf.train.Feature(bytes_list=tf.train.BytesList(value=audio_frame))
        })
      tfwriter.write(new_features.SerializeToString())



