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
import os
import glob
from utils.utils import natural_keys
from collections import defaultdict
from utils.utils import add_directory
from shutil import copyfile

main_dir = "D:/bbt_addon_tree_thesis_2_8" # directory that contains all images and parameters
sketch_dir = os.path.join(main_dir, "material_images")
param_dir = os.path.join(main_dir, "tree_params_NN")

local_dir = os.path.dirname(__file__)
destination_dir = local_dir + "/train_validation_set"
add_directory(destination_dir)
train_dataset = os.path.join(destination_dir, "train_dataset")
add_directory(train_dataset)
train_sketch_dir = os.path.join(train_dataset, "material_images")
add_directory(train_sketch_dir)
train_param_dir = os.path.join(train_dataset, "tree_params_NN")
add_directory(train_param_dir)

val_dataset = os.path.join(destination_dir, "validation_dataset")
add_directory(val_dataset)
val_sketch_dir = os.path.join(val_dataset, "material_images")
add_directory(val_sketch_dir)
val_param_dir = os.path.join(val_dataset, "tree_params_NN")
add_directory(val_param_dir)

list_param_file = glob.glob(os.path.join(param_dir, "*.py"))
list_param_file.sort(key = natural_keys)

tree_type_dict = defaultdict(list)
for file in list_param_file:
    filename_param_split0 = file.split(os.path.sep)
    filename_param_split1 = filename_param_split0[-1].split("_", 1)
    filename_param_split2 = filename_param_split1[1].split(".", 1)
    tree_type = filename_param_split2[0]
    tree_type_dict[tree_type].append(filename_param_split1[0])

list_img_file = glob.glob(os.path.join(sketch_dir, "*.png"))
list_img_file.sort(key = natural_keys)
views = ["front", "back", "left", "right"]


keys_tree_type = tree_type_dict.keys()
print(keys_tree_type)
count_views = 0
for key in keys_tree_type:
    values = tree_type_dict[key]
    for value in values:
        position = int(value)*4
        matching = list_img_file[position:position+4]
        val_img = [s for s in matching if views[count_views] in s]
        for img in val_img:
            filename_param_split0 = img.split(os.path.sep)
            
        train_img = [s for s in matching if views[count_views] not in s]
        for img in train_img:
            filename_param_split0 = img.split(os.path.sep)


    if count_views < len(views)-1:
        count_views += 1
    else:
        count_views = 0
