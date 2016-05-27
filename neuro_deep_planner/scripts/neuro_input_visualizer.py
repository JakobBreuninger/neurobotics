#!/usr/bin/env python

import matplotlib.pyplot as plt
from collections import deque
import rospy
import numpy as np
from nav_msgs.msg import OccupancyGrid


class CostmapVisualizer:

    def __init__(self):

        self.im_data = deque()
        self.im_fig = plt.figure(1)
        self.im_ax = self.im_fig.add_subplot(111)
        self.im_ax.set_title("Visualization of Costmap")
        self.im_im = self.im_ax.imshow(np.zeros((80, 80), dtype='uint8'), cmap=plt.cm.gray, vmin=0, vmax=100,
                                       interpolation="nearest")
        self.im_fig.show()
        self.im_im.axes.figure.canvas.draw()

        self.sub_im = rospy.Subscriber("/move_base/NeuroLocalPlannerWrapper/updated_costmap", OccupancyGrid,
                                       self.im_callback)

    def im_callback(self, data):

        width = data.info.width
        height = data.info.height
        costmap = np.zeros((width, height), dtype='uint8')
        for i in range(width):
            for j in range(height):
                # Somehow we need to change polarity so: 'value' = 100 - 'value'
                costmap[i][j] = 100 - data.data[i + height * j]

        self.im_data.append(costmap)

    def run(self):

        if self.im_data:
            im = self.im_data.popleft()
            self.im_im.set_data(im)
            self.im_im.axes.figure.canvas.draw()


def main():

    rospy.init_node("neuro_input_visualizer", anonymous=False)
    viewer = CostmapVisualizer()
    while not rospy.is_shutdown():
        viewer.run()


if __name__ == '__main__':
    main()
