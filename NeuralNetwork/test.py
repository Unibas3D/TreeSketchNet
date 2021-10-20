# BSD 3-Clause License

# Copyright (c) 2021, ...
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
from dataset.tree_dataset_norm_class import PairImgParamClassification
from utils.utils import natural_keys
import os
import numpy as np 
from models.my_model import ModelTree
from utils.write_logs import read_dictionary_from_file, save_dictionary_to_file
import glob
from utils.my_metrics import OneMinusRMSE
from dataset.parameters_elab import get_subdivision_keys,\
    assign_correct_type_toParams
import ntpath

local_dir = os.path.dirname(__file__)
test_set_dir = local_dir + "/test_set"
log_dir = local_dir + "/logs_archive/fit"

model_name = "alexnet_multiple"
IMG_SHAPE = [608, 608, 3]
if 'resnet' in model_name:
    IMG_SHAPE = [224, 224, 3]
elif 'inception' in model_name:
    IMG_SHAPE = [229, 229, 3]
elif 'alexnet' in model_name:
    IMG_SHAPE = [227, 227, 3]

subdivision_keys = get_subdivision_keys()
list_keys = subdivision_keys.keys()
full_datasets = {}
models = {}
log_dirs = {}

pair_img_param = PairImgParamClassification(IMG_SHAPE, subdivision_keys, \
                            model_name, test_set_dir,\
                            shuffle=False)
pair_img_param.create_dataset(log_dir, write_dataset=False)
full_dataset = pair_img_param.dataset

model = None
print("os.path.exists(log_dir): ", os.path.exists(log_dir))
if os.path.exists(log_dir):
    list_model = [file for file in os.listdir(log_dir) if file.endswith(".h5")]
    print("list_model: ", list_model)
    if list_model:
        list_model.sort(key=natural_keys)
        model = tf.keras.models.load_model(os.path.join(log_dir, list_model[-1]),custom_objects={'OneMinusRMSE':OneMinusRMSE})


list_params = glob.glob(os.path.join(test_set_dir, "tree_params_NN", "*.py"))
list_params.sort(key=natural_keys)

count = 0
dataset_images_paths = pair_img_param.list_images_path
accuracy_dict = {}
for img, labels in full_dataset:
    img_path = str(dataset_images_paths[count])
    head_img_dir, image_filename = ntpath.split(img_path)
    num_img = int(image_filename.split('_')[0])
    params_path = list_params[num_img]
    print("img_path: ", img_path)
    print("params_path: ", params_path)
    respective_dictionary = read_dictionary_from_file(params_path)
    img = tf.expand_dims(img, 0)
    result = model(img)
    loss, loss1, loss2,loss3,\
    loss4, loss5, loss6, \
    one_minus_rmse1, one_minus_rmse2,one_minus_rmse3,\
    one_minus_rmse4, one_minus_rmse5, one_minus_rmse6 = model.evaluate(full_dataset.skip(count).take(1).batch(1), verbose=1)
    print("Restored model, accuracy: {:5.2f}%".format(one_minus_rmse1))
    normalization_matrices = np.load(os.path.join(log_dir, 'normalization_dict.npy'),allow_pickle=True)
    for index,key in enumerate(normalization_matrices.item().keys()):
        denorm = result[index].numpy()*normalization_matrices.item().get(key)
        param_dict = pair_img_param.reconstruct_params_dict(subdivision_keys[key], denorm)

        respective_dictionary.update(param_dict)

    if 'keys_seed' not in normalization_matrices.item().keys():
        respective_dictionary['seed'] = 0

    respective_dictionary = assign_correct_type_toParams(respective_dictionary)

    img_path_split = img_path.split(os.sep)
    folder = os.path.join("output_NN_params", img_path_split[-3])
    save_output_dict_dir = params_path.replace("tree_params_NN", folder)
    save_output_dict_dir = save_output_dict_dir[:len(save_output_dict_dir)-3]
    save_output_dict_dir += "_result.py"
    print(save_output_dict_dir)
    print("----------------------------------")
    head, tail = ntpath.split(save_output_dict_dir)
    if count == 0:
        accuracy_dict_dir = (os.path.sep).join(head.split(os.path.sep)[:-2])
        accuracy_dict_dir = os.path.join(accuracy_dict_dir, "accuracy.py")
    key_acc = head.split(os.path.sep)[-1] + "_" + tail
    accuracy_dict[key_acc] = [one_minus_rmse1, one_minus_rmse2, one_minus_rmse3,\
                        one_minus_rmse4, one_minus_rmse5, one_minus_rmse6]
    if not(os.path.exists(head)):
        os.makedirs(head)

    save_dictionary_to_file(save_output_dict_dir, respective_dictionary)
    count += 1
print(accuracy_dict_dir)
save_dictionary_to_file(accuracy_dict_dir, accuracy_dict)

