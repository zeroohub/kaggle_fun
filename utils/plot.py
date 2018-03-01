# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import itertools
import numpy as np


def plot_images(images, titles=None, columns=4, figsize=(16, 10)):
    fig = plt.figure(figsize=figsize)
    rows = int(len(images) / columns) + 1

    for idx, img in enumerate(images):
        ax = fig.add_subplot(rows, columns, idx + 1)
        if titles is not None:
            ax.set_title(str(titles[idx]))
        plt.imshow(img)
    plt.show()


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def plot_lines(xys, columns=2, figsize=(16, 10)):
    """
    :param xys: [{
            'title': 'accuracy',
            'ys': {
                'train': self.accuracy,
                'valid': self.valid_accuracy,
            },
            "x": self.epoch,
            "xlabel": "epoch",
            "ylabel": "accuracy",

        }, {...}]
    :param columns:
    :param figsize:
    :return:
    """
    fig = plt.figure(figsize=figsize)
    rows = int(len(xys) / columns) + 1
    for idx, xy in enumerate(xys):
        title = xy.get('title', '')
        ys = xy.get('ys', {})
        x = xy.get('x', [])
        xlabel = xy.get('xlabel', "")
        ylabel = xy.get('ylabel', "")
        with_fit = xy.get('fit', True)
        ax = fig.add_subplot(rows, columns, idx + 1)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        for label, y in ys.iteritems():
            ax.plot(x, y, label=label)
            if with_fit:
                fit = np.polyfit(x, y, deg=2)
                x = np.array(x)
                ax.plot(x, fit[0] * np.power(x, 2) + fit[1] * x + fit[2], label='fit-{}'.format(label))

        ax.legend(loc='lower left')
    plt.show()
