#!/usr/bin/env python

import matplotlib.pyplot as plt

from collections import deque

import rospy
from neuro_local_planner_wrapper.msg import Transition

import numpy as np


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

        self.sub_im = rospy.Subscriber("/move_base/NeuroLocalPlannerWrapper/transition", Transition,
                                       self.im_callback)

        self.current_state_rep = Transition()
        print self.current_state_rep

    def im_callback(self, data):

        width = data.width
        height = data.height

        if not data.is_episode_finished:
            data_1d = np.asarray([(100 - data) for data in data.state_representation])

            data_3d = data_1d.reshape(4, 84, 84).swapaxes(1, 2)

            data_3d = np.rollaxis(data_3d, 0, 3)

            divider = np.full((84, 10), 75)

            stacked_costmap = np.hstack((data_3d[:, :, 0], divider,
                                         data_3d[:, :, 1], divider,
                                         data_3d[:, :, 2], divider,
                                         data_3d[:, :, 3]))

            self.im_data.append(stacked_costmap)

    def run(self):

        if self.im_data:
            im = self.im_data.popleft()
            self.im_im.set_data(im)
            self.im_im.axes.figure.canvas.draw()
