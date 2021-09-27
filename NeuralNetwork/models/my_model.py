# BSD 3-Clause License

# Copyright (c) 2021, Nicola Capece, Gilda Manfredi, and Ugo Erra
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import tensorflow as tf
import io
from tensorflow.keras import datasets, layers, models, losses
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.regularizers import l2
from alexnet import AlexNet


class ModelTree(object):

    list_layers_names = []
    base_net_list = ['vgg16', 'mobilenetv2', \
                    'vgg16_multiple', 'resnet50_multiple',\
                    'inception_multiple', 'vgg16_multiple_skip',\
                    'alexnet_multiple']
    summary_dict = {}
    output_layers_names = []

    def __init__(self, _input_shape, _output_shape, base_net):
        self.INPUT_SHAPE = _input_shape
        self.PARAM_SHAPE = _output_shape
        self.summary_dict['input_shape'] = self.INPUT_SHAPE
        self.summary_dict['param_shape'] = self.PARAM_SHAPE
        base_net_l = base_net.lower()
        if base_net_l in self.base_net_list:
            self.summary_dict['base_NN'] = base_net_l
            if base_net_l == self.base_net_list[0]:
                self.vgg16_config()
            elif base_net_l == self.base_net_list[1]:
                self.mobilenetv2_config()
            elif base_net_l == self.base_net_list[2]:
                self.vgg16_multiple_outputs_config()
            elif base_net_l == self.base_net_list[3]:
                self.resnet50_multiple_outputs_config()
            elif base_net_l == self.base_net_list[4]:
                self.inception_multiple_outputs_config()
            elif base_net_l == self.base_net_list[5]:
                self.vgg16_multiple_outputs_connections_config()
            elif base_net_l == self.base_net_list[6]:
                self.alexnet_multiple_outputs_connections_config()


    def mobilenetv2_config(self):
        self.summary_dict['weights'] = 'imagenet'
        self.pretrained_net = MobileNetV2(weights='imagenet',
                    include_top=False,
                    input_shape=(self.INPUT_SHAPE[0], self.INPUT_SHAPE[1], 3))

        for layer in self.pretrained_net.layers:
            self.list_layers_names.append(layer.name)

        layer_names = [
            'block_1_expand_relu',   # 64x64
            'block_3_expand_relu',   # 32x32
            'block_6_expand_relu',   # 16x16
            'block_13_expand_relu',  # 8x8
            'block_16_project',      # 4x4
        ]
        self.summary_dict['base_layers'] = layer_names

        inputs = tf.keras.layers.Input(shape=self.INPUT_SHAPE)
        x = inputs
        base_model = self.cnn_get_some_layers(layer_names)
        all_outputs = base_model(x)
        ''' per 608x608 img
        x = layers.MaxPooling2D((3, 3), strides=(2,2))(all_outputs[-1])
        x = layers.Conv2D(320, 3, strides = (2, 2),activation='relu')(x)
        x = layers.MaxPool2D((3, 3), strides=(1,1))(x)
        x = layers.Conv2D(69, (1,1), activation='relu')(x)
        '''
        ''' per 224x224 img '''
        x = layers.MaxPooling2D((2, 2), strides=(1,1), name="Custom_mp2d")(all_outputs[-1])
        x = layers.Conv2D(320, 3, strides = (1, 1),activation='relu', name = "Custom_conv2d")(x)
        x = layers.MaxPool2D((2, 2), strides=(2,2), name = "Custom_maxPool2d")(x)
        x = layers.Conv2D(69, (1,1), activation='relu', name = "Custom2_conv2d")(x)

        x = layers.Flatten(name = "custom_flatten")(x)
        x = layers.Dense(int(4*69), activation='linear')(x)
        self.summary_dict['output_shape'] = 276

        self.model = tf.keras.Model(inputs=inputs, outputs=x)
        self.model.summary()

        

    def vgg16_config(self):
        self.pretrained_net = VGG16(weights='imagenet',
                    include_top=False,
                    input_shape=(self.INPUT_SHAPE[0], self.INPUT_SHAPE[1], 3))

        for layer in self.pretrained_net.layers:
            self.list_layers_names.append(layer.name)

        taken_layer = "block3_pool"
        inputs = self.pretrained_net.inputs
        x = self.pretrained_net.output
        x = layers.MaxPooling2D((3, 3), strides=(2,2), name="custom_maxpool")(x)
        x = layers.Conv2D(320, 3, strides = (2, 2),activation='relu', name="custom_conv2d")(x)
        x = layers.MaxPool2D((3, 3), strides=(1,1), name="custom_maxpool_2")(x)
        x = layers.Conv2D(69, (1,1), activation='relu', name="custom_conv2d_2")(x)
        x = layers.Flatten(name="custom_flatten")(x)
        if len(self.PARAM_SHAPE) == 1:
            x = layers.Dense(int(self.PARAM_SHAPE[0]), activation='linear', name="dense")(x)
        elif len(self.PARAM_SHAPE) == 2:
            x = layers.Dense(int(self.PARAM_SHAPE[0]*self.PARAM_SHAPE[1]), activation='linear', name="dense")(x)
            x = layers.Reshape((self.PARAM_SHAPE[0], self.PARAM_SHAPE[1]))(x)

        self.model = tf.keras.Model(inputs=inputs, outputs=x)
        self.model.summary()

    def vgg16_multiple_outputs_config(self):
        dropout_dict = {}
        self.pretrained_net = VGG16(weights='imagenet',
                    include_top=False,
                    input_shape=(self.INPUT_SHAPE[0], self.INPUT_SHAPE[1], 3))
        outputs = []
        inputs = self.pretrained_net.inputs
        x = self.pretrained_net.output
        if isinstance(self.PARAM_SHAPE,dict):
            for k, v in self.PARAM_SHAPE.items():
                outputs.append(self.custom_vgg_block(x, v,k, dropout_dict))
            self.model = tf.keras.Model(inputs=inputs, outputs=outputs)
            self.model.summary()
            self.summary_dict['dropout'] = dropout_dict
        else:
            print("self.PARAM_SHAPE is not a dictionary")

    def custom_vgg_block(self, x, param_shape, block_name, dropout_dict):
        
        x = layers.MaxPooling2D((3, 3), strides=(2,2), name="mp_" + block_name)(x)
        x = layers.Conv2D(320, 3, strides = (2, 2),activation='relu', name="conv2d_"+block_name)(x)
        if "-1_1" in block_name or "-inf_inf" in block_name:
            dropout = 0.2
            dropout_dict["dropout_" + block_name] = [dropout]
            x = layers.Dropout(dropout, name="dropout_" + block_name)(x)
        x = layers.MaxPool2D((3, 3), strides=(1,1), name="mp_2_"+block_name)(x)
        x = layers.Conv2D(69, (1,1), activation='relu', name="conv2d_2_"+block_name)(x)
        if "-1_1" in block_name or "-inf_inf" in block_name:
            dropout = 0.2
            dropout_dict["dropout_" + block_name].append(dropout)
            x = layers.Dropout(dropout, name="dropout_2_" + block_name)(x)
        x = layers.Flatten(name="flatten_"+block_name)(x)

        if "-inf_inf" in block_name:
            dropout = 0.2
            dropout_dict["dropout_" + block_name].append(dropout)
            x = layers.Dropout(dropout, name="dropout_3_" + block_name)(x)
        elif "-1_1" in block_name:
            dropout = 0.2
            dropout_dict["dropout_" + block_name].append(dropout)
            x = layers.Dropout(dropout, name="dropout_3_" + block_name)(x)
        elif "angle" in block_name:
            dropout = 0.5
            dropout_dict["dropout_" + block_name] = dropout
            x = layers.Dropout(dropout, name="dropout_" + block_name)(x)
             
        if len(param_shape) == 1:
            x = layers.Dense(int(param_shape[0]), activation='softmax', name="softmax_" + block_name)(x)
        elif len(param_shape) == 2:
            x = layers.Dense(int(param_shape[0]*param_shape[1]), activation='linear', name="linear_"+block_name)(x)
            x = layers.Reshape((param_shape[0], param_shape[1]), name= "reshape_" + block_name)(x)
        x = layers.Lambda(lambda x: tf.identity(x, name=block_name), name=block_name)(x)
        self.output_layers_names.append(x.name)
        return x

    def vgg16_multiple_outputs_connections_config(self):
        dropout_dict = {}
        self.pretrained_net = VGG16(weights='imagenet',
                    include_top=False,
                    input_shape=(self.INPUT_SHAPE[0], self.INPUT_SHAPE[1], 3))
        outputs = []
        inputs = self.pretrained_net.inputs
        x = self.pretrained_net.output
        block1_pool = self.pretrained_net.get_layer("block1_pool").output
        block2_pool = self.pretrained_net.get_layer("block2_pool").output

        skip_b2_b6 = layers.Conv2D(512, (4, 4), strides = (4,4), activation='relu', name="skip_b2_b6_conv1")(block2_pool)
        skip_b2_b6 = layers.MaxPool2D((2, 2), strides=(2, 2), name="skip_b2_b6_pool")(skip_b2_b6)

        skip_b1_b7 = layers.Conv2D(128, (10, 10), strides = (8,8), activation='relu', name="skip_b1_b7_conv1")(block1_pool)
        skip_b1_b7 = layers.MaxPool2D((4, 4), strides=(4, 4), name="skip_b1_b7_pool")(skip_b1_b7)

        x = layers.Add()([x, skip_b2_b6])
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name="block6_conv1")(x)
        x = layers.MaxPool2D((2, 2), strides=(2, 2), name="block6_pool")(x)
        x = layers.Add()([x, skip_b1_b7])
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name="block7_conv1")(x)
        x = layers.MaxPool2D((2, 2), strides=(2, 2), name="block7_pool")(x)
        x = layers.Flatten(name="flatten_custom")(x)
        if isinstance(self.PARAM_SHAPE,dict):
            for k, v in self.PARAM_SHAPE.items():
                outputs.append(self.custom_vgg_conn_block(x, v,k, dropout_dict))
            self.model = tf.keras.Model(inputs=inputs, outputs=outputs)
            self.model.summary()
            self.summary_dict['dropout'] = dropout_dict
        else:
            print("self.PARAM_SHAPE is not a dictionary")

    def custom_vgg_conn_block(self, x, param_shape, block_name, dropout_dict):

        if "-inf_inf" in block_name or "0_1" in block_name:
            dropout = 0.2
            dropout_dict["dropout_" + block_name]= dropout
            x = layers.Dropout(dropout, name="dropout_3_" + block_name)(x)
        elif "angle" in block_name or "sigmoid" in block_name:
            dropout = 0.5
            dropout_dict["dropout_" + block_name] = dropout
            x = layers.Dropout(dropout, name="dropout_3_" + block_name)(x)
             
        if len(param_shape) == 1:
            if 'sigmoid' in block_name:
                x = layers.Dense(int(param_shape[0]), activation='sigmoid', name="sigmoid_" + block_name)(x)
        elif len(param_shape) == 2:
            x = layers.Dense(int(param_shape[0]*param_shape[1]), activation='linear', name="linear_"+block_name)(x)
            x = layers.Reshape((param_shape[0], param_shape[1]), name= "reshape_" + block_name)(x)
        x = layers.Lambda(lambda x: tf.identity(x, name=block_name), name=block_name)(x)
        self.output_layers_names.append(x.name)
        return x

    def alexnet_multiple_outputs_connections_config(self):
        dropout_dict = {}
        outputs = []
        self.pretrained_net = AlexNet((self.INPUT_SHAPE[0], self.INPUT_SHAPE[1],3))
        inputs = self.pretrained_net.inputs
        x = self.pretrained_net.output
        if isinstance(self.PARAM_SHAPE,dict):
            for k, v in self.PARAM_SHAPE.items():
                outputs.append(self.custom_alexnet_conn_block(x, v,k, dropout_dict))
            self.model = tf.keras.Model(inputs=inputs, outputs=outputs)
            self.model.summary()
            self.summary_dict['dropout'] = dropout_dict
        else:
            print("self.PARAM_SHAPE is not a dictionary")

    def custom_alexnet_conn_block(self, x, param_shape, block_name, dropout_dict):
             
        if len(param_shape) == 1:
            if 'sigmoid' in block_name:
                x = layers.Dense(int(param_shape[0]), activation='sigmoid', name="sigmoid_" + block_name)(x)
        elif len(param_shape) == 2:
            x = layers.Dense(int(param_shape[0]*param_shape[1]), activation='linear', name="linear_"+block_name)(x)
            x = layers.Reshape((param_shape[0], param_shape[1]), name= "reshape_" + block_name)(x)
        x = layers.Lambda(lambda x: tf.identity(x, name=block_name), name=block_name)(x)
        self.output_layers_names.append(x.name)
        return x

    def resnet50_multiple_outputs_config(self):
        dropout_dict = {}
        self.pretrained_net = ResNet50(weights='imagenet',
                                    include_top=False,
                                    input_shape=(self.INPUT_SHAPE[0], self.INPUT_SHAPE[1], 3))
        outputs = []
        inputs = self.pretrained_net.inputs
        x = self.pretrained_net.output
        x = layers.Flatten(name="flatten_gen")(x)
        x = layers.Dropout(0.2, name="dropout_gen_1")(x)
        x = layers.Dense(4000, activation='linear', name="linear_gen_1")(x)
        x = layers.Dropout(0.2, name="dropout_gen_2")(x)
        x = layers.Dense(2000, activation='linear', name="linear_gen_2")(x)
        x = layers.Dropout(0.2, name="dropout_gen_3")(x)
        x = layers.Dense(1000, activation='linear', name="linear_gen_3")(x)
        x = layers.Dropout(0.2, name="dropout_gen_4")(x)
        x = layers.Dense(512, activation='linear', name="linear_gen_4")(x)
        x = layers.Dropout(0.2, name="dropout_gen_5")(x)
        if isinstance(self.PARAM_SHAPE,dict):
            for k, v in self.PARAM_SHAPE.items():
                outputs.append(self.custom_resnet50_block(x, v,k, dropout_dict))
            self.model = tf.keras.Model(inputs=inputs, outputs=outputs)
            self.model.summary()
        else:
            print("self.PARAM_SHAPE is not a dictionary")

    def custom_resnet50_block(self, x, param_shape, block_name, dropout_dict):
             
        if len(param_shape) == 1:
            x = layers.Dense(int(param_shape[0]), activation='softmax', name="softmax_" + block_name)(x)
        elif len(param_shape) == 2:
            x = layers.Dense(int(param_shape[0]*param_shape[1]), activation='linear', name="linear_"+block_name)(x)
            x = layers.Reshape((param_shape[0], param_shape[1]), name= "reshape_" + block_name)(x)
        x = layers.Lambda(lambda x: tf.identity(x, name=block_name), name=block_name)(x)
        self.output_layers_names.append(x.name)
        return x

    def inception_multiple_outputs_config(self):
        dropout_dict = {}
        self.pretrained_net = InceptionV3(weights='imagenet',
                                    include_top=False,
                                    input_shape=(self.INPUT_SHAPE[0], self.INPUT_SHAPE[1], 3))
        outputs = []
        inputs = self.pretrained_net.inputs
        x = self.pretrained_net.get_layer("mixed4").output
        x = layers.Flatten(name="flatten_gen")(x)
        x = layers.Dense(4000, activation='linear', name="linear_gen_1")(x)
        x = layers.Dense(2000, activation='linear', name="linear_gen_2")(x)
        if isinstance(self.PARAM_SHAPE,dict):
            for k, v in self.PARAM_SHAPE.items():
                outputs.append(self.custom_inception_block(x, v,k, dropout_dict))
            self.model = tf.keras.Model(inputs=inputs, outputs=outputs)
            self.model.summary()
        else:
            print("self.PARAM_SHAPE is not a dictionary")

    def custom_inception_block(self, x, param_shape, block_name, dropout_dict):
        if "0_1" in block_name or "-inf_inf" in block_name:
            x = layers.Dense(1000, activation='linear', kernel_regularizer=l2(1e-2), name="linear_3_" + block_name)(x)
            x = layers.Dense(512, activation='linear', kernel_regularizer=l2(1e-2), name="linear_4_" + block_name)(x)
        else:
            x = layers.Dense(1000, activation='linear', name="linear_3_" + block_name)(x)
            x = layers.Dense(512, activation='linear', name="linear_4_" + block_name)(x)

        if len(param_shape) == 1:
            if 'sigmoid' in block_name:
                x = layers.Dense(int(param_shape[0]), activation='sigmoid', name="sigmoid_" + block_name)(x)
        elif len(param_shape) == 2:
            if "0_1" in block_name or "-inf_inf" in block_name:
                x = layers.Dropout(0.2, name="dropout_4_" + block_name)(x)
            elif "angle" in block_name:
                x = layers.Dropout(0.5, name="dropout_4_" + block_name)(x)
            x = layers.Dense(int(param_shape[0]*param_shape[1]), activation='linear', name="linear_5_"+block_name)(x)
            x = layers.Reshape((param_shape[0], param_shape[1]), name= "reshape_" + block_name)(x)
        x = layers.Lambda(lambda x: tf.identity(x, name=block_name), name=block_name)(x)
        self.output_layers_names.append(x.name)
        return x

    def cnn_get_output_layer(self, layer_name):
        self.pretrained_net.trainable = False
        return self.pretrained_net.get_layer(layer_name).output


    def cnn_trainable_some_layers(self, layer_name):
        '''CNN without top layer
            Make trainable only some layers 
            from first layer to index '''
        layer_i = self.list_layers_names.index(layer_name)
        self.pretrained_net.trainable = True
        for layer in self.pretrained_net.layers[:(layer_i)]:
            print(layer.name)

    def cnn_get_some_layers(self, layer_names):
        layers = [self.pretrained_net.get_layer(name).output for name in layer_names]
        down_stack = tf.keras.Model(inputs=self.pretrained_net.input, outputs=layers)
        return down_stack

    def get_model_summary(self, model):
        stream = io.StringIO()
        model.summary(print_fn=lambda x: stream.write(x + '\n'))
        summary_string = stream.getvalue()
        stream.close()
        return summary_string


    def custom_vgg16(self, _input_shape, _output_shape):
        """
        Create model based on VGG16
        
        Arguments:
        _input_shape -- list [img_width, img_height, 1] that represent a
                        grayscale image dimension
        _output_shape -- list [4, num_parameters] of tree's parameters
        
        Returns: 
        model
        """ 
        conv_base = VGG16(weights='imagenet',
                    include_top=False,
                    input_shape=(_input_shape[0], _input_shape[1], 3))

        conv_base.summary()

        model = models.Sequential()
        if _input_shape[2] == 1:
            model.add(layers.Input(shape=_input_shape, dtype='float32'))
            model.add(layers.BatchNormalization())
            model.add(layers.Conv2D(10, kernel_size = (1,1), padding = 'same', activation = 'relu'))
            model.add(layers.Conv2D(3, kernel_size = (1,1), padding = 'same', activation = 'relu'))
        model.add(conv_base)
        model.add(layers.Flatten())
        model.add(layers.Dense(int(_output_shape[0]*_output_shape[1]), activation='linear'))

        conv_base.trainable = False

    def custom_vgg16_ed(self, _input_shape, _output_shape):
        """
        Create model based on VGG16
        
        Arguments:
        _input_shape -- list [img_width, img_height, 3] that represent a
                        RGB image dimension
        _output_shape -- list [4, num_parameters] of tree's parameters
        
        Returns: 
        model
        """ 
        inputs = layers.Input(shape=(_input_shape[0], _input_shape[1], 3))
        vgg16 = VGG16(weights='imagenet',
                    include_top=False,
                    input_tensor=inputs)
        vgg16.summary

        return tf.keras.Model(inputs=inputs, outputs=vgg16, name='custom_vgg16_ed')