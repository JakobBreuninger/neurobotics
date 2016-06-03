#!/usr/bin/env python

import matplotlib.pyplot as plt
from collections import deque
import rospy
import numpy as np
from nav_msgs.msg import OccupancyGrid


class CostmapVisualizer:

    def __init__(self):

        self.im_data = deque()
        self.im_fig = plt.figure(1, figsize=(40, 5))
        self.im_ax = self.im_fig.add_subplot(111)
        self.im_im = self.im_ax.imshow(np.zeros((80, 320), dtype='uint8'), cmap=plt.cm.gray, vmin=0, vmax=100,
                                       interpolation="nearest", )

        self.im_fig.show()
        plt.axis('off')
        self.im_im.axes.figure.canvas.draw()

        self.sub_im = rospy.Subscriber("/move_base/NeuroLocalPlannerWrapper/constcutive_costmaps", OccupancyGrid,
                                       self.im_callback)

    def im_callback(self, data):

        width = data.info.width
        height = data.info.height

        fake_data = np.asarray([(100 - data) for data in data.data])
        # fake_data = np.hstack((fake_data, fake_data, fake_data, fake_data))

        data_3d = fake_data.reshape(4, 80, 80).swapaxes(1, 2)

        divider = np.full((80, 10), 75)

        stacked_costmap = np.hstack((data_3d[0], divider, data_3d[1], divider, data_3d[2], divider, data_3d[3]))

        self.im_data.append(stacked_costmap)

    def run(self):

        if self.im_data:
            im = self.im_data.popleft()
            self.im_im.set_data(im)
            self.im_im.axes.figure.canvas.draw()
