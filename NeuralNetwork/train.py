# BSD 3-Clause License

# Copyright (c) 2022, Gilda Manfredi, Nicola Capece, and Ugo Erra
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
import numpy as np
import tensorflow as tf
from tensorflow import keras
from models.my_model import ModelTree
import os
from dataset.tree_dataset_norm_class import PairImgParamClassification
from utils.write_logs import write_text_logs, write_img_logs_to_dir
from utils.utils import natural_keys
from utils.utils import StopCallback, CustomTQDMProgressBar
from dataset.parameters_elab import get_subdivision_keys
from utils.my_metrics import OneMinusRMSE
from utils.write_logs import save_dictionary_to_file

local_dir = os.path.dirname(__file__)
dataset_dir = local_dir + "/train_validation_set"
log_dir = local_dir + "/logs_archive/fit"
load_dir = local_dir + "/logs_archive/fit"
if not os.path.exists(log_dir):
        os.makedirs(log_dir)

IMG_SHAPE = [608, 608, 3]
BATCH_SIZE = 8
model_name = 'efficientnet_multiple'

if 'resnet' in model_name or 'efficientnet' in model_name:
    IMG_SHAPE = [224, 224, 3]
elif 'inception' in model_name:
    IMG_SHAPE = [229, 229, 3]
elif 'alexnet' in model_name:
    IMG_SHAPE = [227, 227, 3]
elif 'coatnet' in model_name:
    IMG_SHAPE = [224, 224, 3]

train = True
save_callback = True
load_model = False

subdivision_keys = get_subdivision_keys()
print(subdivision_keys)


train_dataset_path = os.path.join(dataset_dir, "train_dataset")
pair_img_param_train = PairImgParamClassification(IMG_SHAPE, subdivision_keys, model_name, train_dataset_path)
pair_img_param_train.create_dataset(log_dir, "train_dataset")
train_dataset = pair_img_param_train.dataset

val_dataset_path = os.path.join(dataset_dir, "validation_dataset")
pair_img_param_val = PairImgParamClassification(IMG_SHAPE, subdivision_keys, model_name, val_dataset_path)
pair_img_param_val.create_dataset(log_dir, "validation_dataset")
validation_dataset = pair_img_param_val.dataset

summary_dict = pair_img_param_train.summary_dict

normalization_dict = pair_img_param_train.normalization_dict
np.save(os.path.join(log_dir, 'normalization_dict.npy'), normalization_dict)


summary_dict['batch_size'] = BATCH_SIZE
summary_dict['subdivision_keys'] = subdivision_keys

train_dataset = train_dataset.batch(BATCH_SIZE)
validation_dataset = validation_dataset.batch(BATCH_SIZE)

TRAIN_CARDINALITY = pair_img_param_train.cardinality
VAL_CARDINALITY = pair_img_param_val.cardinality

print("TRAIN_CARDINALITY: ", TRAIN_CARDINALITY)
STEPS_PER_EPOCH = np.ceil(TRAIN_CARDINALITY/BATCH_SIZE)
STEPS_PER_EPOCH_VAL = np.ceil(VAL_CARDINALITY/BATCH_SIZE)
summary_dict['train_dataset_cardinality'] = TRAIN_CARDINALITY
print(train_dataset.element_spec)

for image, labels in train_dataset.take(1):
    print(image.shape)


LEARNING_RATE = 1e-5
optimizer = keras.optimizers.Adam(learning_rate= LEARNING_RATE)
summary_dict['optimizer']= "adam with learning_rate = " + str(LEARNING_RATE)

metrics = [OneMinusRMSE(), tf.keras.metrics.RootMeanSquaredError()]
metrics_names = [i._name for i in metrics]
summary_dict['metrics']=metrics_names
summary_dict['loss_function']='mean_squared_error'
subnet_losses = {}
subnet_metrics = {}

for k in pair_img_param_train.params_shapes_dict.keys():
    if "classification" not in k:
        subnet_losses[k] = 'mean_squared_error'
        subnet_metrics[k] = OneMinusRMSE()
    else:
        subnet_losses[k] = 'categorical_crossentropy'
        subnet_metrics[k] = 'accuracy'


model = None
if load_model == False:
    model_obj = ModelTree(IMG_SHAPE, pair_img_param_train.params_shapes_dict, model_name)
    model = model_obj.model
    print("output_layers_names: ", model_obj.output_layers_names)
    summary_dict.update(model_obj.summary_dict)
    model.compile(optimizer=optimizer, loss=subnet_losses, metrics=subnet_metrics)
else:
    print("os.path.exists(load_dir): ", os.path.exists(load_dir))
    if os.path.exists(load_dir):
        list_model = [file for file in os.listdir(load_dir) if file.endswith(".h5")]
        print("list_model: ", list_model)
        if list_model:
            list_model.sort(key=natural_keys)
            model = tf.keras.models.load_model(os.path.join(load_dir, list_model[-1]),custom_objects={'OneMinusRMSE':OneMinusRMSE})


EPOCHS = 2000
summary_dict['epochs']=EPOCHS

save_dictionary_to_file(os.path.join(log_dir, "summary_dict.py"), summary_dict)

if save_callback is True:
    model._layers = [layer for layer in model._layers if isinstance(layer, keras.layers.Layer)]
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(log_dir, 'model_{epoch:02d}.h5'), period = 70)
    stop_callback = StopCallback(log_dir)
    custom_progbar = CustomTQDMProgressBar()
    reduceLR = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=30, mode='auto', verbose=1)


    write_text_logs(summary_dict, log_dir)
    write_img_logs_to_dir(model, log_dir)

if train is True:
    model.fit(
        train_dataset,
        epochs=EPOCHS,
        verbose = 0,
        validation_data=validation_dataset,
        callbacks=[custom_progbar,tensorboard_callback, model_checkpoint, stop_callback])
