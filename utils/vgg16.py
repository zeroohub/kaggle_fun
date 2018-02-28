from __future__ import division, print_function

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from functools import partial
from keras.utils.data_utils import get_file
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.layers.convolutional import MaxPooling2D, ZeroPadding2D, Conv2D


vgg_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32).reshape((3, 1, 1))


def vgg_preprocess(x):
    """
        Subtracts the mean RGB value, and transposes RGB to BGR.
        The mean RGB was computed on the image set used to train the VGG model.

        Args: 
            x: Image array (height x width x channels)
        Returns:
            Image array (height x width x transposed_channels)
    """
    x = x - vgg_mean
    return x[:, ::-1]  # reverse axis rgb->bgr


def build_conv(model, layers, filters):
    for i in range(layers):
        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(filters, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))


def build_fc(model, units, dropout=0.5):
    model.add(Dense(units, activation='relu'))
    model.add(Dropout(dropout))


def get_model(weight='imagenet'):
    model = Sequential()
    model.add(Lambda(vgg_preprocess, input_shape=(3, 224, 224), output_shape=(3, 224, 224)))

    build_conv(model, 2, 64)
    build_conv(model, 2, 128)
    build_conv(model, 3, 256)
    build_conv(model, 3, 512)
    build_conv(model, 3, 512)

    model.add(Flatten())
    build_fc(model, 4096)
    build_fc(model, 4096)
    model.add(Dense(1000, activation='softmax'))

    if weight == 'imagenet':
        fname = 'vgg16.h5'
        model.load_weights(get_file(fname, 'http://files.fast.ai/models/{}'.format(fname), cache_subdir='models'))

    return model

