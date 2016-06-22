#!/usr/bin/env python

import matplotlib.pyplot as plt

from collections import deque

import numpy as np


class CostmapVisualizer:

    def __init__(self):

        self.im_data = deque()
        self.im_fig = plt.figure(1, figsize=(40, 40))
        self.im_ax = self.im_fig.add_subplot(111)

        self.im_fig.show()
        plt.axis('off')

    def set_data(self, data_3d):

        self.im_im = self.im_ax.imshow(np.zeros(((len(data_3d[0])+10)*len(data_3d)+10, len(data_3d[0])*4+30),
                                                dtype='uint8'), cmap=plt.cm.gray, vmin=0, vmax=100,
                                       interpolation="nearest", )

        h_divider = np.full((data_3d.shape[1], 10), 75)
        v_divider = np.full((10, data_3d.shape[1]*4+30), 75)

        output = v_divider

        for data in data_3d:

            h_stack = np.hstack((data[:, :, 0], h_divider,
                                 data[:, :, 1], h_divider,
                                 data[:, :, 2], h_divider,
                                 data[:, :, 3]))

            v_stack = np.vstack((h_stack, v_divider))

            output = np.vstack((output, v_stack))

        self.im_data.append(output)

    def run(self):

        if self.im_data:
            im = self.im_data.popleft()
            self.im_im.set_data(im)
            self.im_im.axes.figure.canvas.draw()
