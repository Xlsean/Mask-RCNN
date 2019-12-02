########################################################################


########################################################################

import re
import os
import cv2
import time
import numpy as np

from PIL import Image, ImageChops

import torch
import torch.nn as nn

from torch.autograd import Variable

########################################################################


def time_counter(func):
    def wrapper(*args, **kwargs):
        t1 = time.time()
        result = func(*args, **kwargs)
        t2 = time.time() - t1
        print('Took {0:.4} sec on "{1}" func'.format(t2, func.__name__))
        return result
    return wrapper


def slice_array(array, batch_size):
    '''
    To slice an array into some certain size of pieces

    + array: an array waiting to be taken apart into pieces (np.array)
    '''
    return np.array([array[i:i + batch_size]
                     for i in range(0, array.shape[0], batch_size)])


def get_file_path(train_ratio, data_dir, extension='.JPG|.png|.jpeg'):
    '''
    To get all the targetted file path and split them into training 
    and testing set based on "train_ratio" ratio
    '''
    target = re.compile('^[^\.].*(' + extension + ')$')
    path_list = []
    for path, folders, files in os.walk(data_dir):
        for file in files:
            try:
                full_name = target.match(file).group()
                full_path = os.path.join(path, full_name)
                path_list.append(full_path)
            except:
                pass

    # To convert the python list into numpy array.
    path_list = np.array(path_list)
    if train_ratio < 1:
        train_list, test_list = slice_array(
            path_list, int(path_list.shape[0] * train_ratio))
    else:
        train_ratio = 1
        train_list = slice_array(
            path_list, int(path_list.shape[0] * train_ratio))[0]
        test_list = np.array([])
    return train_list, test_list


def one_hot(labels, num_classes=10):
    '''
    The input "labels" shape can be (batch, label_number) 2D array, or
    the shape can simply be 1D array with corresponding label numbers.
    '''
    return np.eye(num_classes, dtype=float)[labels]


def weights_init_normal(layer):
    '''
    Here are all listed init methods provided by Pytorch framework.
    Link: https://pytorch.org/docs/stable/_modules/torch/nn/init.html
    '''
    class_name = layer.__class__.__name__

    if class_name.find('Conv') != -1:
        nn.init.normal_(layer.weight.data, 0.0, 0.02)

    elif class_name.find('BatchNorm2d') != -1:
        nn.init.normal_(layer.weight.data, 1.0, 0.02)
        nn.init.constant_(layer.bias.data, 0.0)


def compute_ap(recall, precision):
    '''
    Compute 'average precision' by giving the recall and precision curve.
    Both recall and precision curve are 'list' format.
    '''

    # Append sentinel values at the end.
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # Compute the precision envelop.
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # To calculate area under PR curve, looking for points where
    # x axis (recall) changes value.
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def get_test_input(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (416, 416))

    # cv2 read images as "BGR". Turn "BGR" to "RGB".
    # torch sequence is (B, C, H, W), transpose the tensor.
    img_ = img[:, :, ::-1].transpose((2, 0, 1))

    # Expand a dim at 0 and normalize the data
    img_ = img_[np.newaxis, :, :, :] / 255.0
    img_ = torch.from_numpy(img_).float()
    img_ = Variable(img_)

    return img_


def trim(img):
    bg = Image.new(img.mode, img.size, img.getpixel((0, 0)))
    diff = ImageChops.difference(img, bg)
    diff = ImageChops.add(diff, diff, scale=2.0, offset=-100)
    bbox = diff.getbbox()
    return img.crop(bbox)


########################################################################
# For testing the functions defined above.

path = '/Users/kcl/Documents/Python_Projects/01_AI_Tutorials/04_Tensorflow_Series/Code_Session/TF_12_YOLO_v3/cfg/yolov3.cfg'


if __name__ == '__main__':
    a = model_config(path)

    print(a)
