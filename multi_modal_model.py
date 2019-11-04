import tensorflow as tf
from tf_Data_Builder import tf_Data_Builder
import os
import sys
import matplotlib.pyplot as plt
import numpy as np

data_dir = '/Users/lukeprice/Codes/MultiMode/data/Youtube8M/'
""" dir for new clipped tfrecords """
outDir = data_dir + 'Y8M_build/'
""" clipped files for model training """
trainFiles = [os.path.join(outDir, file) for file in os.listdir(outDir) if '.tfrecord' in file]

tfd = tf_Data_Builder(
  batchsize = 32, 
  target_classes = np.array([[170],[1454],[709],[1057],[1308]]))
IDs, labels, images, audio = tfd.create_dataset(tf_files = trainFiles)




# fig, ax = plt.subplots(4,3)
# print(labels[0])
# ax[0,0].imshow(tf.reshape(images[0,0], [32,32]))
# ax[0,1].imshow(tf.reshape(images[0,150], [32,32]))
# ax[0,2].imshow(tf.reshape(images[0,-1], [32,32]))
# print(labels[3])
# ax[1,0].imshow(tf.reshape(images[3,0], [32,32]))
# ax[1,1].imshow(tf.reshape(images[3,150], [32,32]))
# ax[1,2].imshow(tf.reshape(images[3,-1], [32,32]))
# print(labels[1])
# ax[2,0].imshow(tf.reshape(images[1,0], [32,32]))
# ax[2,1].imshow(tf.reshape(images[1,150], [32,32]))
# ax[2,2].imshow(tf.reshape(images[1,-1], [32,32]))
# print(labels[2])
# ax[3,0].imshow(tf.reshape(images[2,0], [32,32]))
# ax[3,1].imshow(tf.reshape(images[2,150], [32,32]))
# ax[3,2].imshow(tf.reshape(images[2,-1], [32,32]))
# plt.show()



sys.exit()


# def play_one_vid(record_name, video_index):
#   """ record_name = list of IDs
#       video_index = any arbitrary index in IDs list """
#     return vid_ids[video_index]
    
# # this worked on my local jupyter notebook, but doesn't show on kaggle kernels:
# """ YouTubeVideo requires from IPython.display import YouTubeVideo """
# YouTubeVideo(play_one_vid(video_lvl_record, 7))


