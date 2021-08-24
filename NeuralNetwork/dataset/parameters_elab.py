import ast
import os
from collections import defaultdict
import numpy as np

def convert_leaf_shape_int(value):
    if value == 'rect':
        return 1
    elif value == 'hex':
        return 0

def convert_leaf_shape_str(value):
    if value >= 0.5:
        return 'rect'
    elif value < 0.5:
        return 'hex'

def convert_choiceNegPos_0_1(value):
    if value == 1:
        return 1
    elif value == -1:
        return 0

def convert_choiceNegPos_minus1_1(value):
    if value >= 0.5:
        return 1
    elif value < 0.5:
        return -1

def importDataFromDir(filename):
        settings = {}
        try:

            file = open(filename, "r")
            contents = file.read()
            settings = ast.literal_eval(contents)
            file.close()

        except (FileNotFoundError, IOError):
            print("File Not Found")
        return settings

def get_all_keys():
    unused_keys = ['prune', 'pruneBase', 'prunePowerHigh', 'prunePowerLow',
                        'pruneRatio', 'pruneWidth','pruneWidthPeak']

    local_dir = os.path.dirname(__file__)
    dictionary = importDataFromDir(os.path.join(local_dir, "parameter_converter_dict.py"))
    keys = dictionary.keys()
    all_keys = defaultdict(list)

    for key in keys:
        if not(key in unused_keys):
            all_keys['keys_all'].append(key)
    return all_keys

def get_subdivision_keys():
    unused_keys = ['prune', 'pruneBase', 'prunePowerHigh', 'prunePowerLow',
                        'pruneRatio', 'pruneWidth','pruneWidthPeak', 'seed']

    local_dir = os.path.dirname(__file__)
    dictionary = importDataFromDir(os.path.join(local_dir, "parameter_converter_dict.py"))
    keys = dictionary.keys()
    subdivision_keys = defaultdict(list)

    for key in keys:
        if not(key in unused_keys):
            value = dictionary.get(key)
            if key == "seed":
               subdivision_keys['keys_seed'].append(key) 
            elif value[0] == 0.0 and value[1] == 1.0:
                subdivision_keys['keys_0_1'].append(key)

            elif "angle" in key.lower() or "rotate" in key.lower() or\
                ("curve" in key.lower() and key != "curveRes"):
                subdivision_keys['keys_angle'].append(key)

            elif value[0] == 99999999999999999 and value[1] == -99999999999999999:
                subdivision_keys['keys_-inf_inf'].append(key)

            elif value[0] == 0 and value[1] == -99999999999999999:
                subdivision_keys['keys_0_inf'].append(key)

            elif value[0] == -1.0 and value[1] == 1.0:
                subdivision_keys['keys_-1_1'].append(key)

            else:
                subdivision_keys['keys_min_max'].append(key)
    return subdivision_keys


def round_value(key, value):
    local_dir = os.path.dirname(__file__)
    param_converter = importDataFromDir(os.path.join(local_dir, "parameter_converter_dict_2.py"))
    value_converter_dict = param_converter[key]
    max_val = value_converter_dict[1]
    round_value = round(value)
    if value > max_val and round_value <= max_val:
        value = round_value
    return value

def value_from_list(key, value):
    list_shapes = {}
    list_shapes['shape'] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10]
    list_shapes['shapeS'] = [0, 1, 2, 3, 4, 5, 6, 7, 10]
    list_shapes['leafDist'] = [0, 1, 2, 3, 4, 5, 6, 7, 10]
    round_value = int(round(value))
    min_before_val = 11
    max_after_val = 0
    for num in list_shapes[key]:
        if num == round_value:
            return str(num)
        else:
            if num < round_value:
                min_before_val = num
            elif num > round_value:
                max_after_val = num
    mean = float(min_before_val + max_after_val)/2.0
    if round_value <= mean:
        return str(min_before_val)
    else:
        return str(max_after_val)


def assign_correct_type_toParams(dictionary):
    key_int = ['baseSplits', 'bevelRes', 'branches', 
                'curveRes', 'leaves', 'levels', 'maxBaseSplits',
                'resU', 'seed', 'nrings']
    key_minus1_1_only = ['choiceNegPos1', 'choiceNegPos2']
    key_bool = ['horzLeaves', 'useOldDownAngle', 'useParentAngle']
    key_char_int = ['leafDist', 'shape', 'shapeS']
    key_round_1 = ['customShape', 'taper', 'branchDist']
    key_leaf_shape = ['leafShape']
    for key in dictionary.keys():
        value = dictionary[key]
        if key in key_int:
            if isinstance(value, list):
                dictionary[key] = [int(round(i)) for i in value]
            else:
                dictionary[key] = int(round(value))
        elif key in key_char_int:
            dictionary[key] = value_from_list(key, value)
        elif key in key_minus1_1_only:
            dictionary[key] = convert_choiceNegPos_minus1_1(value)
        elif key in key_bool:
            if isinstance(value, list):
                dictionary[key] = [True if i >= 0.5 else False for i in value]
            else:
                dictionary[key] = True if value >= 0.5 else False
        elif key in key_round_1: 
            if isinstance(value, list):
                dictionary[key] = [round_value(key, i) for i in value]
            else:
                dictionary[key] = round_value(key, value)
        elif key in key_leaf_shape:
            dictionary[key] = convert_leaf_shape_str(value)
        elif not(isinstance(value, str)):
            if isinstance(value, list):
                dictionary[key] = [round(i, 2) for i in value]
            else:
                dictionary[key] = round(value, 2)

    return dictionary


def adjust_min_max(dictionary):
    local_dir = os.path.dirname(__file__)
    param_converter = importDataFromDir(os.path.join(local_dir, "parameter_converter_dict_2.py"))
    for key in dictionary.keys():
        value_converter_dict = param_converter[key]
        min_val = value_converter_dict[0]
        max_val = value_converter_dict[1]

        if isinstance(dictionary[key], list):
            dictionary[key] = [min(max_val, i) for i in dictionary[key]]
            dictionary[key] = [max(min_val, i) for i in dictionary[key]]
        else:
            if not(isinstance(dictionary[key], str)):
                dictionary[key] = min(max_val, dictionary[key])
                dictionary[key] = max(min_val, dictionary[key])
    return dictionary
            