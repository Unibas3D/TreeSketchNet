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
import tensorflow as tf
import os
import collections
import numpy as np
from pathlib import Path
import ast
import glob
from math import inf
from utils import natural_keys
from normalizations import choose_normalization
from parameters_elab import convert_leaf_shape_int, convert_choiceNegPos_0_1
from collections import defaultdict

some_keys = ['attractOut', 'baseSize', 'baseSize_s', 'horzLeaves', 'leafScaleV', 
            'lengthV', 'prune', 'pruneBase', 'pruneRatio', 'rSplits2', 'radiusTweak', 
            'scaleV0', 'splitHeight', 'taper', 'taperCrown', 'useOldDownAngle', 'useParentAngle']

class PairImgParamClassification(object):
    summary_dict = {}
    labels_dict = {}
    normalization_dict = {}
    list_classes_str = []
    params_shapes_dict = {}
    params_types_dict = {}
    def __init__(self, image_shape, _subdivision_keys, _NN_model_name,\
                main_dir = "", shuffle = True):
        self.IMG_SHAPE = image_shape
        self.NN_model_name = _NN_model_name

        param_dir = os.path.join(main_dir, "tree_params_NN")

        sketch_dir = os.path.join(main_dir, "styles_sketch_images")
        if os.path.exists(sketch_dir):
            p = Path(sketch_dir)
            print(str(p))
            self.list_ds = tf.data.Dataset.list_files(str(p/'*/*.png'), shuffle = shuffle)
        else:
            sketch_dir = os.path.join(main_dir, "sketch_images")
            p = Path(sketch_dir)
            print(str(p))
            self.list_ds = tf.data.Dataset.list_files(str(p/'*.png'), shuffle = shuffle)
        self.list_images_path = list(self.list_ds.as_numpy_iterator())
        self.cardinality = len(self.list_images_path)
        for filename in self.list_images_path[0:2]:
            print(filename)
        print("----------------------------------------------------------")
        self.list_param_file = glob.glob(os.path.join(param_dir, "*.py"))
        self.list_param_file.sort(key = natural_keys)
        self.param_file_structure_path = self.list_param_file[0]
        for filename in self.list_param_file[:2]:
            print(filename)
        for filename in self.list_param_file[-2:]:
            print(filename)
        print("----------------------------------------------------------")
        self.create_params_matrices(self.list_param_file, _subdivision_keys)
        print(self.list_classes_str)
        for key, val in self.labels_dict.items():
            print(key + ": " + str(val.shape))

        for k in self.labels_dict.keys():
            if "collection" in k:
                self.params_types_dict[k] = tf.uint8
            else:
                self.params_types_dict[k] = tf.float32

    def create_dataset(self, directory, dataset_name = 'full_dataset', \
                            num_file = -1, write_dataset = True):
        if num_file > 0 and num_file < tf.data.experimental.cardinality(self.list_ds).numpy():
            self.list_ds = self.list_ds.take(int(num_file))
        if write_dataset is True:
            writer = tf.data.experimental.TFRecordWriter(os.path.join(directory, dataset_name + '.tfrecord'))
            writer.write(self.list_ds)
        self.dataset = self.list_ds.map(self.map_func, \
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
        self.dataset = self.dataset.map(self.map_func_dict, \
        num_parallel_calls=tf.data.experimental.AUTOTUNE)

    def load_dataset(self, directory, dataset_name = 'full_dataset'):
        path = os.path.join(directory, dataset_name + '.tfrecord')
        self.list_ds = tf.data.TFRecordDataset(filenames = [path])
        for filename in self.list_ds.take(2):
            print(filename)
        print("----------------------------------------------------------")
        self.num_samples = self.list_ds.reduce(np.int64(0), lambda x, _: x + 1).numpy()
        self.dataset = self.list_ds.map(self.map_func, \
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
        self.dataset = self.dataset.map(self.map_func_dict, \
        num_parallel_calls=tf.data.experimental.AUTOTUNE)

    def generator(self):
        for filename in self.list_ds:
            image = self.tensor_images(filename)
            labels = self.tensor_parameters(filename)
            yield image, labels

    def map_func(self, key):
        result_tensors = tf.py_function(func=self.map_func_py, \
                                        inp=[key], \
                                        Tout=[tf.float32] + \
                                            list(self.params_types_dict.values()))
        result_tensors[0].set_shape(self.IMG_SHAPE)
        for i,k in enumerate(self.params_shapes_dict.keys(), start=1):
            result_tensors[i].set_shape(self.params_shapes_dict[k])

        return result_tensors

    def map_func_dict(self, *item):
        dict_result = {}
        for i, k in enumerate(self.labels_dict.keys(), start=1):
            dict_result[k] = item[i]
        return item[0], dict_result

    def map_func_py(self, filename):
        image = self.tensor_images(filename)
        labels = self.tensor_parameters(filename)
        return [image] + list(labels.values())

    def tensor_images(self, filename):
        image = tf.io.read_file(filename)
        image = tf.io.decode_png(image, channels=self.IMG_SHAPE[2])

        image = tf.image.resize(image, [self.IMG_SHAPE[0], self.IMG_SHAPE[1]])
        image = tf.cast(image, tf.float32)
        image = image.numpy()
        if 'inception' in self.NN_model_name:
            ''' image normalized btw -1 and 1 '''
            image = (image / 127.5) - 1.0
        else:
            ''' image normalized btw 0 and 1 '''
            image = image/255.0
            ''' image normalized btw -1 and 1 '''
        return image

    def get_params_index_from_img(self, filename):
        parts = tf.strings.split(filename, sep=os.path.sep)
        file_name = parts[-1]
        filename_parts = tf.strings.split(file_name, sep="_")
        num_str = filename_parts[0]
        num = tf.strings.to_number(num_str, tf.int32)
        return num.numpy()

    def tensor_parameters(self, filename):
        index = self.get_params_index_from_img(filename)
        labels = {}
        for key, val in self.labels_dict.items():
            labels[key] = val[index]
        return labels  

    def create_params_matrices(self, list_param_file, subdivision_keys):
        m = len(list_param_file)
        first_file = list_param_file[0]
        print("first_file: ", first_file)
        self.elab_data(self.importDataFromDir(first_file),subdivision_keys,m,0)
        for index in range(1, m):
            file = list_param_file[index]
            self.elab_data(self.importDataFromDir(file),subdivision_keys,m,index)
        for norm_type in subdivision_keys.keys():
            _, normalization_matrix, result = choose_normalization(norm_type, self.labels_dict[norm_type])
            self.normalization_dict[norm_type] = normalization_matrix
            self.labels_dict[norm_type] = result
        
    def importDataFromDir(self, filename):
        settings = {}
        try:

            file = open(filename, "r")
            contents = file.read()
            settings = ast.literal_eval(contents)
            file.close()

        except (FileNotFoundError, IOError):
            print("File Not Found")
        return settings

    def elab_data(self, dictionary, subdivision_keys, m, i):
        keys = dictionary.keys()
        for norm_type, some_keys in subdivision_keys.items():
            nw_nh = len(some_keys)
            nc = 4
            i_sample = np.zeros([nw_nh, nc], dtype=np.float32)
            index = 0
            for key in keys:
                if key in some_keys:
                    value = dictionary.get(key)
                    if key == 'leafShape':
                        value = convert_leaf_shape_int(value)
                    elif 'choiceNegPos' in key:
                        value = convert_choiceNegPos_0_1(value)
                    if isinstance(value, list) and len(value) == 4:
                        i_sample[index] = [float(i) for i in value]
                    else:
                        value = float(value)
                        i_sample[index] = [value, value, value, value]
                    index = index + 1
            result = np.transpose(i_sample)
            if i == 0:
                self.labels_dict[norm_type] = np.zeros([m] + list(result.shape), dtype=np.float32)
                self.params_shapes_dict[norm_type] = result.shape
            self.labels_dict[norm_type][i] = result

    def label_classes(self, filename, m, index):
        filename_split_slash = os.path.split(filename)
        filename_split = filename_split_slash[1].split('_')
        class_str = ""
        for i in range(1, len(filename_split)):
            s = filename_split[i]
            if i == len(filename_split)-1:
                s_split = s.split('.')
                class_str += s_split[0]
            else:
                class_str += s + '_'

        if class_str not in self.list_classes_str:
            self.list_classes_str.append(class_str)
        result = self.list_classes_str.index(class_str)
        if index == 0:
             self.labels_dict['classification_label'] = np.zeros([m], dtype=np.uint8)
        self.labels_dict['classification_label'][index] = result

    def get_classes(self, filename):
        filename_split_slash = os.path.split(filename)
        filename_split = filename_split_slash[1].split('_')
        class_str = ""
        for i in range(1, len(filename_split)):
            s = filename_split[i]
            if i == len(filename_split)-1:
                s_split = s.split('.')
                class_str += s_split[0]
            else:
                class_str += s + '_'

        if class_str not in self.list_classes_str:
            self.list_classes_str.append(class_str)
        return self.list_classes_str.index(class_str)


    def reconstruct_params_dict(self, list_keys, NN_output):
        result_dictionary = {}
        param_struct = self.importDataFromDir(self.param_file_structure_path)
        NN_output = np.transpose(np.squeeze(NN_output, axis=0))
        count = 0
        for key in list_keys:
            value = param_struct[key]
            if isinstance(value, list) and len(value) == 4:
                result_dictionary[key] = list(NN_output[count])
            else:
                result_dictionary[key] = NN_output[count][0]
            count += 1
            
        return result_dictionary

    def get_param_dict_from_img(self, filename):
        index = self.get_params_index_from_img(filename)
        return self.list_param_file[index]


