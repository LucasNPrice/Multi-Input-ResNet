import tensorflow as tf
from tqdm import tqdm
import os 
import numpy as np
from random import shuffle

class tf_Record_Processor():
  def __init__(self):
    pass

  def write_segments(self, inDir, outDir):
    inDirFiles = os.listdir(inDir)
    new_Examples = []
    outfiles = np.arange(0, len(inDirFiles))
    outfiles = ['F' + str(i) for i in outfiles]
    outfiles = [i + '.tfrecord' for i in outfiles]
    for file in inDirFiles:
      if '.tfrecord' in file:
        outfile = os.path.join(outDir, file)
        raw_tfrecord = os.path.join(inDir, file)
        dataset = tf.data.TFRecordDataset(raw_tfrecord)
        for raw_example in dataset:
          new_Examples += self.__split_on_frames(raw_example)
    # shuffle new examples to ensure equal distribution of classes across files 
    shuffle(new_Examples)
    for example in new_Examples:
      file = np.random.choice(outfiles) 
      file = os.path.join(outDir, file)
      tfwriter = tf.io.TFRecordWriter(file)
      tfwriter.write(example.SerializeToString())

  def __split_on_frames(self, raw_example):
    segmented = []
    example = tf.train.SequenceExample.FromString(raw_example.numpy())
    image = example.feature_lists.feature_list['rgb']
    audio = example.feature_lists.feature_list['audio']
    n_frames = len(image.feature)
    for i in range (0, n_frames):
      img_frame = image.feature[i].bytes_list.value
      audio_frame = audio.feature[i].bytes_list.value
      if any(i != 0 for i in audio_frame) or any(i != 0 for i in img_frame):
      # tf.train.Feature(bytes_list=tf.train.BytesList(value=img_frame))
        new_features = tf.train.Features(feature={
          'id': example.context.feature['id'],
          'frame': tf.train.Feature(int64_list=tf.train.Int64List(value=[i])),
          'labels': example.context.feature['labels'],
          'rgb': tf.train.Feature(bytes_list=tf.train.BytesList(value=img_frame)),
          'audio': tf.train.Feature(bytes_list=tf.train.BytesList(value=audio_frame))
          })
        new_example = tf.train.Example(features=new_features)
        segmented.append(new_example)
      # tfwriter.write(new_example.SerializeToString())
    return segmented

  def read_segmented_tfrecord(self, raw_tfrecord):
    dataset = tf.data.TFRecordDataset(raw_tfrecord)
    for raw_record in dataset.take(1):
      example = tf.train.Example()
      example.ParseFromString(raw_record.numpy())
      print(example)
