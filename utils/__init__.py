# -*- coding: utf-8 -*-
from keras.preprocessing import image


def get_accuracy(model, batches):
    return model.evaluate_generator(batches, batches.samples)[1]


def get_batches(path, gen=image.ImageDataGenerator(), target_size=(224, 224), class_mode='categorical', **kwargs):
    return gen.flow_from_directory(path, target_size=target_size,
                                   class_mode=class_mode, **kwargs)
