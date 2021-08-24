import numpy as np

def abs_max_normalization(matrix):
    matrix_abs = np.absolute(matrix)
    max_1 = np.max(matrix_abs, axis=1, keepdims=True)
    max_0 = np.max(max_1, axis=0, keepdims=True)
    result = matrix / max_0
    return max_0, result

def abs_max_normalization_0_division(matrix):
    matrix_abs = np.absolute(matrix)
    max_1 = np.max(matrix_abs, axis=1, keepdims=True)
    max_0 = np.max(max_1, axis=0, keepdims=True)
    result = np.divide(matrix, max_0, out=np.zeros(matrix.shape, dtype=float), where=max_0!=0)
    return max_0, result

def angle_normalization(matrix):
    shape_matrix = list(matrix.shape)
    max_0 = np.ones((1, 1, shape_matrix[-1])) * 360.0
    result = matrix / max_0
    return max_0, result

def no_normalization(matrix):
    shape_matrix = list(matrix.shape)
    max_0 = np.ones((1, 1, shape_matrix[-1]))
    return max_0, matrix

def choose_normalization(key, matrix):
    if key == 'keys_angle':
        print(key, ": angle normalization")
        type_norm = "angle normalization"
        norm_matrix, result = angle_normalization(matrix)
        return type_norm, norm_matrix, result
    elif key == 'keys_0_1' or key == 'keys_-1_1' or key == 'keys_sigmoid':
        print(key,": no normalization")
        type_norm = "no normalization"
        norm_matrix, result = no_normalization(matrix)
        return type_norm, norm_matrix, result
    elif key == 'keys_all':
        print(key,": abs max normalization 0 division")
        type_norm = "abs max normalization 0 division"
        norm_matrix, result = abs_max_normalization_0_division(matrix)
        return type_norm, norm_matrix, result
    else:
        print(key,": abs max normalization")
        type_norm = "abs max normalization"
        norm_matrix, result = abs_max_normalization(matrix)
        return type_norm, norm_matrix, result

sampl = np.random.uniform(low=-20.0, high=20, size=(2, 4, 5))
angle_normalization(sampl)