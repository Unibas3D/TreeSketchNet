import tensorflow as tf
import numpy as np
import datetime
import os
import ast
import matplotlib.pyplot as plt
import pprint

def write_img_logs(img, img_shape):
    img = np.reshape(img, (-1, img_shape[0], img_shape[1], img_shape[2]))
    local_dir = os.path.dirname(__file__)
    logdir = local_dir + "/logs/images" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    file_writer = tf.summary.create_file_writer(logdir)
    with file_writer.as_default():
        tf.summary.image("Training data", img, step=0)

def write_img_logs_to_dir(model, dir):
    path = os.path.join(dir,"model.png")
    tf.keras.utils.plot_model(model, to_file=path, show_shapes=True,dpi=64)
    image = tf.io.read_file(path)
    image = tf.io.decode_png(image, channels=1)
    image = tf.cast(image, tf.float32)
    img = np.expand_dims(image.numpy(), 0)
    file_writer = tf.summary.create_file_writer(dir)
    with file_writer.as_default():
        tf.summary.image("Model Summary", img, step=0)

def write_text_logs(dictionary, dir):
    file_writer = tf.summary.create_file_writer(dir)
    with file_writer.as_default():
        out_vec = np.array(["", ""])
        for key, value in dictionary.items():
            if key == 'base_layers':
                value1 = "  \n".join(value)
            else:
                value1 = str(value)
            vec = np.array(["**" + key + "**", value1])
            out_vec = np.vstack((out_vec, vec))
        tf.summary.text("Procedure summary", tf.convert_to_tensor(out_vec), step=0)

def write_array_on_file(path, data, filename = 'normalization_matrix.txt', fmt = "%f" ):
    path = os.path.join(path, filename)
    outfile= open(path, 'w')
    outfile.write('# Array shape: {0}\n'.format(data.shape))

    print("len data sh: ", len(data.shape))
    if len(data.shape) <= 1:
        np.savetxt(outfile, data, fmt = fmt)
    else:
        for data_slice in data:
            np.savetxt(outfile, data_slice, fmt=fmt)

            outfile.write('# New slice\n')
    outfile.close()

def read_array_from_file(path, filename = 'normalization_matrix.txt', dtype = float):
    path = os.path.join(path, filename)
    infile = open(path, 'r')
    first_line = infile.readline()
    line_split = first_line.split(sep=": ")
    original_shape_str = line_split[1]
    original_shape_str = original_shape_str[1:len(original_shape_str)-2]
    original_shape_split = original_shape_str.split(sep=",")
    original_shape_list = [int(i.strip()) for i in original_shape_split if i]
    return np.loadtxt(path, dtype = dtype).reshape(original_shape_list)

def save_dictionary_to_file(path, dictionary):
    f = open(path,"w")
    f.write(pprint.pformat(dictionary))
    f.close()

def read_dictionary_from_file(path):
        settings = {}
        try:

            file = open(path, "r")
            contents = file.read()
            settings = ast.literal_eval(contents)
            file.close()

        except (FileNotFoundError, IOError):
            print("File Not Found")
        return settings
