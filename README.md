# Tree Sketch to 3D tree meshes
![teaser](https://user-images.githubusercontent.com/88141714/127903685-1d0aa283-2ecb-4cc0-9ccd-a436e3e60aaa.jpg)

## Overview
TreeSketchNet is an automated procedural modelling system based on Deep Convolutional Neural Network which is described in our preprint [paper](). 
This system allows to create 3D tree meshes by predicting the Weber and Penn<sup>([1](#References))</sup> parameters using simple hand drawn sketches. In particular this system can recognize and create 3D mesh

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
### Neural Network
In [NeuralNetwork](NeuralNetwork) folder you can find the code related to the neural network architecture.  
* You can download the dataset used in our work from [here](). You need to unpack the .zip file and copy its content in the [NeuralNetwork/train_validation_set](NeuralNetwork/train_validation_set) folder.  
You can create your own dataset using the Blender add-on [Render Tree Thesis]().
* To test our pre-trained network you can download the model form [here](), copy the entire folder content in [NeuralNetwork/logs_archive](NeuralNetwork/logs_archive) and execute the `test.py` file.
* You can find all the network architectures tested in this paper in [my_model.py](NeuralNetwork/models/my_model.py) file. To train one of these networks, in `train.py` you need to set the `model_name` variable, choosing from the strings listed below:
  * `resnet50_multiple`: ResNet50
  * `inception_multiple`: InceptionNet V3
  * `vgg16_multiple_skip`: VGG-16 with skip connections
  * `alexnet_multiple`: AlexNet
* You can also change the directory in which to save the trained model by editing the `log_dir` variable in the `train.py` file. Remember that if you need to test your trained network you have to change the same variable in the `test.py` file!
* You can early stop the training from terminal with `Ctrl + C`. The procedure will save the model with best weights.
* To continue the training of an early stopped model, in the `train.py` file you must
  * change the `load_dir` variable with the directory where the saved model is located;
  * set the `load_model` variable to True.

  Remember that the saved model architecture must be the same as that indicated by the `model_name` variable.
<inserire indicazioni per il download del dataset e del modello. Inserire indicazioni sulla struttura della cartella di test>
 
### Blender add-on Render Tree Thesis
 For the installation of the Render Tree Thesis add-on you need to follow these steps:
 1. In the installation folder of Blender 2.82 (for example `C:\Program Files\Blender Foundation\Blender 2.82\`), open `addons` folder following this path `Blender 2.82\2.82\scripts\addons\`. Download the [add_curve_sapling.zip](add_curve_sapling.zip) file, unzip it and paste the `add_curve_sapling` folder in the `addons` directory replacing the existing files.
 2. Enable the Sapling Tree Gen add-on as shown in the following images  
![sapling_addon_1](imgs/sapling_addon_1.png)  ![sapling_addon_2](imgs/sapling_addon_2.png)
 3. Download [addon_render_tree_thesis.zip]() file.
 4. In `Blender Preferences -> Add-ons` click on `Install...` and selected the .zip file downloaded in point 2. After the installation enable the Render Tree Thesis add-on by checking the box to the left of its name.  
 ![sapling_addon_3](imgs/sapling_addon_3.png)
 5. After the installation, you can find the add-on in the right panel of the 3D Viewport.  
 ![sapling_addon_4](imgs/sapling_addon_4.png)

## Media
...

### Video
...

### Citation
...

## License
Copyright (c) 2021, Nicola Capece. All right reserved.
 
The code is distributed under a BSD license. See `LICENSE` for information.
 
### References
1. __Creation and Rendering of Realistic Trees.__ 
   Jason Weber and Joseph Penn. 1995. *InProceedings of the 22nd Annual Conference on Computer Graphics and InteractiveTechniques (SIGGRAPH ’95).* Association for Computing
   Machinery, New York, NY,USA, 119–128. [link](https://doir.org/10.1145/218380.218427)
