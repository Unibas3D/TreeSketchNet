# Tree Sketch to 3D tree meshes
![teaser](https://user-images.githubusercontent.com/88141714/127903685-1d0aa283-2ecb-4cc0-9ccd-a436e3e60aaa.jpg)

## Overview
...

## Specification
* Python 3.7.4
* Tensorflow 2.2.0, tensorflow-addons, Cuda 10.1
* Numpy
* Blender 2.82

For Blender installation click on [this link](https://download.blender.org/release/Blender2.82/).  
For Tensorflow and Cuda installation follow [this guide](https://www.tensorflow.org/install/pip).  
After installing tensorflow, you can install numpy and tensorflow-addons via the Python **pip** package manager, as follows:
```
$ pip install numpy tensorflow-addons
```

## Usage
In [NeuralNetwork](NeuralNetwork) folder you can find the code related to the neural network architecture.  
* You can download the dataset used in our work from [here]() or you can create your own using the Blender add-on [RenderTreeThesis]().
* To test our pre-trained network you can download the model form [here](), copy the entire folder content in [NeuralNetwork/logs_archive](NeuralNetwork/logs_archive) and execute the `test.py` file.
* You can find all the network architectures tested in this paper in [my_model.py](NeuralNetwork/models/my_model.py) file. To train one of these networks, in `train.py` you must set the `model_name` variable, choosing bethween the strings listed below:
  * `resnet50_multiple`: ResNet50
  * `inception_multiple`: InceptionNet V3
  * `vgg16_multiple_skip`: VGG-16 with skip connections
  * `alexnet_multiple`: AlexNet
* You can also change the directory where to save the trained model, modifing the `log_dir` variable in the `train.py` file. Remember that if you need to test your trained network you must modify the same variable in the `test.py` file!
* You can early stop the training from terminal with `Ctrl + C`. The procedure will save the model with best weights.
* To continue the training of an early stopped model, in the `train.py` file you must
  * change the `load_dir` variable with the directory where the saved model is;
  * set the `load_model` variable to True.
  Remember that the saved model architecture needs to be the same of that indicated by the `model_name` variable.
<inserire indicazioni per il download del dataset e del modello. Inserire indicazioni sulla struttura della cartella di test>

## Media
...

### Video
...

### Citation
...

## License
...
