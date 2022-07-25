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
