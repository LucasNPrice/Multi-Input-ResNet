# Multi-Input ResNet

**multi-input.py**
- multi-input neural net (2D image and 1D audio data)
- calls ResNet.py for 1D and 2D residual network processing 

**ResNet.py**
  - residual network class for 2D image data and 1D (audio) data 
  - can be used to train model end to end or for transfer methods with head/tail layers removed. 

**segmented_data_builder.py**
  - builds TensorFlow Dataset for dynamic loading of data for models 
  - includes embdedding/multi-hot methods for label transformations 
  
**focal_loss.py**
    - multi-class focal loss function 
    - in this case, alpha is an array where each element is the weighting factor for that specific class
        - for example, if classifying 5 different classes then alpha should be a (5,1) array of weights 
    - dramatic increase in performance over Binary Crosstropy 
  
  ## Network Graph
<!--  ![Image description](multi_model.png =500x2000)-->
  <img src="multi_model.png" alt="model_graph" width="500" height="1500"/>

