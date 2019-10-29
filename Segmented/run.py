import tensorflow as tf
from tf_Data_Builder import tf_Data_Builder
# from tfRecord_Processor import tfRecord_Processor
from tfRecord_Processor_2 import tf_Record_Processor
import os
import sys

data_dir = '/Users/lukeprice/github/multi-modal/datafiles/Y8M_build/'
trainFiles = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if '.tfrecord' in file]

tfr = tf_Record_Processor()
tfr.write_segments(inDir = data_dir, 
  outDir = '/Users/lukeprice/github/multi-modal/datafiles/Y8M_segmented')
