import tensorflow as tf
# from tfRecord_Processor import tfRecord_Processor
from tfRecord_Processor_2 import tf_Record_Processor
import os
import sys

data_dir = '/Users/lukeprice/github/multi-modal/datafiles/Y8M_build/'
trainFiles = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if '.tfrecord' in file]

tfr = tf_Record_Processor()
tfr.write_segments(inDir = data_dir, 
  outDir = '/Users/lukeprice/github/multi-modal/datafiles/Y8M_segmented')

segmented_dir = '/Users/lukeprice/github/multi-modal/datafiles/Y8M_segmented/'
read_file = os.path.join(segmented_dir, os.listdir(segmented_dir)[0])
tfr.read_segmented_tfrecord(read_file)