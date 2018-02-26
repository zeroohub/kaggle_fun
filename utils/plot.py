# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt


def plots(images, columns=4):
    fig = plt.figure(figsize=(20, 10))
    rows = int(len(images) / columns) + 1

    for idx, img in enumerate(images):
        fig.add_subplot(rows, columns, idx + 1)
        plt.imshow(img)
    plt.show()