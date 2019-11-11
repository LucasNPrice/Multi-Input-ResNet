import matplotlib.pyplot as plt
import pickle
import numpy as np
import sys

def get_metrics_from_file(pickle_file):

  metric_history =[]
  with open(pickle_file, 'rb') as file:
    logged_metrics = pickle.load(file)
  Epochs = logged_metrics['Epochs']

  for epoch in Epochs:
    metric_history.append(Epochs[epoch]['metrics']['F1'])

  return metric_history

def plot_metrics(*args, **kwargs):

  max_num_epochs = np.max([len(arr) for arr in args])

  if len(kwargs.keys()) != 0:
    if 'names' in kwargs.keys():
      names = kwargs['names']
      assert len(names) == len(args)
      for i, arg in enumerate(args):
        print(names[i])
        plt.plot(arg, label=names[i])
      plt.legend(loc='upper left')
    else:
      for arg in args:
        plt.plot(arg)

  plt.xticks(np.arange(0, max_num_epochs, 1))
  plt.xlabel('Epochs', fontsize=10)
  plt.ylabel('F1-score', fontsize=10)
  plt.savefig('learning_comparisons.png')
  plt.show()


if __name__ == '__main__':
  # multi_modal_weights was written over
  # multi_modal_metrics was incorrectly written two extra metrics 
  
  multi_metric_file = '/Users/lukeprice/github/multi-modal/metric_files/multi_modal_metrics.pickle'
  image_metric_file = '/Users/lukeprice/github/multi-modal/metric_files/image_only_metrics.pickle'
  audio_metric_file = '/Users/lukeprice/github/multi-modal/metric_files/audio_only_metrics.pickle'
  
  multi_metrics = get_metrics_from_file(multi_metric_file)[:-1]
  image_metrics = get_metrics_from_file(image_metric_file)
  audio_metrics = get_metrics_from_file(audio_metric_file)
  
  plot_metrics(multi_metrics, image_metrics, audio_metrics, 
               names=['multi-modal', 'image-only', 'audio-only'])
