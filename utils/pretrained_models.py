# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from functools import partial

from keras.applications.vgg16 import WEIGHTS_PATH, get_file
from keras.models import Sequential
from keras.layers import Conv2D, InputLayer, MaxPooling2D, Dense, Flatten, Lambda, Dropout
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing.image import ImageDataGenerator


class VGG16(object):

    def __init__(self):
        super(VGG16, self).__init__()
        self.model = Sequential()
        self.model.add(InputLayer((224, 224, 3)))
        self.model.add(Lambda(preprocess_input))
        self.build_conv_blocks()
        self.build_fc_blocks()
        self.model.load_weights(get_file('vgg16_weights_tf_dim_ordering_tf_kernels.h5',
                                         WEIGHTS_PATH,
                                         cache_subdir='models',
                                         file_hash='64373286793e3c8b2b4e3219cbf3544b'))

    def build_conv_blocks(self):
        self.add_conv(2, 64)
        self.add_conv(2, 128)
        self.add_conv(3, 256)
        self.add_conv(3, 512)
        self.add_conv(3, 512)

    def add_conv(self, layers, filters):
        for i in range(layers):
            self.model.add(Conv2D(filters, (3, 3), activation='relu', padding='same'))
        self.model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    def add_dense(self, units, activation='relu', dropout=0.5):
        self.model.add(Dense(units, activation=activation))
        if dropout != 0:
            self.model.add(Dropout(0.5))


    def build_fc_blocks(self):
        self.model.add(Flatten())
        self.add_dense(4096)
        self.add_dense(4096)
        self.add_dense(1000, 'softmax', 0)

    @classmethod
    def get_model(cls, units=1000):
        vgg = cls()
        vgg.model.pop()
        vgg.add_dense(units, 'softmax', 0)
        return vgg

    @staticmethod
    def get_data_gen(directory, batch_size=32, gen=None):
        if not gen:
            gen = ImageDataGenerator()
        gen.flow_from_directory = partial(gen.flow_from_directory,
                                          directory=directory,
                                          class_mode='categorical',
                                          target_size=(224, 224),
                                          shuffle=False,
                                          batch_size=batch_size)
        return gen
