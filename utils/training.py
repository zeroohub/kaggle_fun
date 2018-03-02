# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from functools import partial

from keras.callbacks import Callback, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator

from .plot import plot_lines


class TrainHistory(Callback):

    def on_train_begin(self, logs={}):
        self.epoch = []
        self.loss = []
        self.valid_loss = []
        self.accuracy = []
        self.valid_accuracy = []

    def on_epoch_begin(self, epoch, logs=None):
        self.batch = []
        self.batch_loss = []
        self.batch_accuracy = []

    def on_batch_end(self, batch, logs={}):
        self.batch.append(batch)
        self.batch_loss.append(logs.get('loss', -1))
        self.batch_accuracy.append(logs.get('acc', -1))

    def on_epoch_end(self, epoch, logs={}):
        self.epoch.append(epoch)
        self.loss.append(logs.get('loss', -1))
        self.valid_loss.append(logs.get('val_loss', -1))
        self.accuracy.append(logs.get('acc', -1))
        self.valid_accuracy.append(logs.get('val_acc', -1))

        xys = [{
            'title': 'loss',
            'ys': {
                'train': self.batch_loss,
            },
            "x": self.batch,
            "xlabel": "batch",
            "ylabel": "loss",

        }, {
            'title': 'accuracy',
            'ys': {
                'train': self.batch_accuracy,
            },
            "x": self.batch,
            "xlabel": "batch",
            "ylabel": "accuracy",

        }]
        plot_lines(xys)

    def on_train_end(self, logs=None):
        xys = [{
            'title': 'loss',
            'ys': {
                'train': self.loss,
                'valid': self.valid_loss,
            },
            "x": self.epoch,
            "xlabel": "epoch",
            "ylabel": "loss",
            'fit': False,

        }, {
            'title': 'accuracy',
            'ys': {
                'train': self.accuracy,
                'valid': self.valid_accuracy,
            },
            "x": self.epoch,
            "xlabel": "epoch",
            "ylabel": "accuracy",
            'fit': False,

        }]
        plot_lines(xys)


def fit_generator(model, train_flow, valid_flow, epochs=3, callbacks=None):
    default_callbacks = [TrainHistory(),
                         EarlyStopping(monitor='val_loss',
                                       min_delta=0.002,
                                       patience=2,
                                       verbose=1)]
    if callbacks:
        default_callbacks += callbacks

    model.fit_generator(
        train_flow,
        steps_per_epoch=getattr(train_flow, 'samples', train_flow.x.shape[0]) // train_flow.batch_size,
        epochs=epochs,
        validation_data=valid_flow,
        validation_steps=getattr(valid_flow, 'samples', valid_flow.x.shape[0]) // valid_flow.batch_size,
        callbacks=default_callbacks
    )


def wrap_data_gen(directory=None, batch_size=32, gen=None):
    if not gen:
        gen = ImageDataGenerator()
    gen.flow_from_directory = partial(gen.flow_from_directory,
                                      directory=directory,
                                      class_mode='categorical',
                                      target_size=(224, 224),
                                      shuffle=False,
                                      batch_size=batch_size)

    gen.flow = partial(gen.flow,
                       shuffle=False,
                       batch_size=batch_size)

    return gen
