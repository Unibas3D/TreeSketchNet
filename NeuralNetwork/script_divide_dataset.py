import os
import glob
from utils import natural_keys
from collections import defaultdict
from utils import add_directory
from shutil import copyfile

main_dir = "D:/Gilda Manfredi/bbt_addon_tree_thesis_2_8"
# sketch_dir = os.path.join(main_dir, "sketch_images")
sketch_dir = os.path.join(main_dir, "material_images")
param_dir = os.path.join(main_dir, "tree_params_NN")

destination_dir = "D:/Gilda Manfredi/division_dataset"
add_directory(destination_dir)
train_dataset = os.path.join(destination_dir, "train_dataset")
add_directory(train_dataset)
# train_sketch_dir = os.path.join(train_dataset, "sketch_images")
train_sketch_dir = os.path.join(train_dataset, "material_images")
add_directory(train_sketch_dir)
train_param_dir = os.path.join(train_dataset, "tree_params_NN")
add_directory(train_param_dir)

val_dataset = os.path.join(destination_dir, "validation_dataset")
add_directory(val_dataset)
# val_sketch_dir = os.path.join(val_dataset, "sketch_images")
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
    # copyfile(file, os.path.join(train_param_dir, filename_param_split0[-1]))
    # copyfile(file, os.path.join(val_param_dir, filename_param_split0[-1]))

list_img_file = glob.glob(os.path.join(sketch_dir, "*.png"))
list_img_file.sort(key = natural_keys)
views = ["front", "back", "left", "right"]


keys_tree_type = tree_type_dict.keys()
print(keys_tree_type)
count_views = 0
for key in keys_tree_type:
    values = tree_type_dict[key]
    for value in values:
        # matching = [s for s in list_img_file if value in s]
        position = int(value)*4
        matching = list_img_file[position:position+4]
        # print(matching)
        val_img = [s for s in matching if views[count_views] in s]
        # print(val_img)
        for img in val_img:
            filename_param_split0 = img.split(os.path.sep)
            # copyfile(img, os.path.join(val_sketch_dir, filename_param_split0[-1]))
            
        train_img = [s for s in matching if views[count_views] not in s]
        # print(train_img)
        for img in train_img:
            filename_param_split0 = img.split(os.path.sep)
            # copyfile(img, os.path.join(train_sketch_dir, filename_param_split0[-1]))


    if count_views < len(views)-1:
        count_views += 1
    else:
        count_views = 0
