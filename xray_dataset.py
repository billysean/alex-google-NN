from __future__ import absolute_import, print_function

import os
import sys
from six.moves import urllib
import tarfile

import numpy as np
import pickle
import cv2
from data_utils import *


def load_data(dirname="dataset", resize_pics=(224, 224), shuffle=True,
    one_hot=False):
    dataset_file = os.path.join(dirname, 'grayscale_227.pkl')
   
    X, Y = build_image_dataset_from_dir(os.path.join(dirname, 'xray/'),
                                        dataset_file=dataset_file,
                                        resize=resize_pics,
                                        filetypes=['.jpg', '.jpeg', '.png'],
                                        convert_gray=False,
                                        shuffle_data=shuffle,
                                        categorical_Y=one_hot)


    X = np.expand_dims (X,axis = -1)
    print(X.shape)
    return X, Y

def load_single_image(dirname):
    image = cv2.imread(dirname)
    print("original : ",image.shape)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print("GRAY : ",image.shape)
    image = cv2.resize(image,(227,227))
    print("GRAY-RESIZED : ",image.shape)
    # image = cv2.fastNlMeansDenoising(image,None,10,7,21)
    return image

if __name__ == '__main__':
    X = load_single_image('./TB1.jpg')
    cv2.imshow('new',X)
    cv2.waitKey(0)