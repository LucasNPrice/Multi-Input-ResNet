# multi-modal

## Multi-Input ResNet

**ResNet.py**
  - residual network class for 2D image data 
  - can be used to train model end to end or for transfer methods with head/tail layers removed. 

**multi-input.py**
  - multi-input neural net (2D image and 1D audio data)
  - calls ResNet.py for 2D image residual network 

**segmented_data_builder.py**
  - builds TensorFlow Dataset for dynamic loading of data for models 
  - includes embdedding/multi-hot methods for label transformations 
  
<!--  ![Image description](multi_model.png =500x2000)-->
  <img src="multi_model.png" alt="model_graph" width="500" height="1000"/>

