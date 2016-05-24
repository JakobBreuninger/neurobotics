#!/usr/bin/env python

import rospy
from nav_msgs.msg import OccupancyGrid
import numpy as np
import matplotlib.pyplot as plt


def callback(data):
    rospy.loginfo("Success")

    # Get width and height of the occupancy grid map
    width = data.info.width
    height = data.info.height

    # Initialize the array
    costmap = [[0 for x in range(width)] for y in range(height)]

    # Transform msg into array
    for i in range(width):
        for j in range(height):
            costmap[i][j] = data.data[i + height * j]

    imgplot = plt.imshow(costmap)

    plt.show(block=True)


def main():

    rospy.init_node("neuro_deep_planner", anonymous=False)

    rospy.Subscriber("/move_base/NeuroLocalPlannerWrapper/updated_costmap", OccupancyGrid, callback)

    rospy.spin()


if __name__ == '__main__':
    main()

import matplotlib.pyplot as plt
from matplotlib import cm
from cv_bridge import CvBridge, CvBridgeError
import cv
from collections import deque
import rospy
from sensor_msgs.msg import Image
import numpy as np

class CameraViewer():

    def __init__( self, root='navbot' ):

        self.root = root
        self.im_data = deque()
        self.bridge = CvBridge() # For converting ROS images into a readable format

        self.im_fig = plt.figure( 1 )
        self.im_ax = self.im_fig.add_subplot(111)
        self.im_ax.set_title("DVS Image")
        self.im_im = self.im_ax.imshow( np.zeros( ( 256, 256 ),dtype='uint8' ), cmap=plt.cm.gray ) # Blank starting image
        #self.im_im = self.im_ax.imshow( np.zeros( ( 256, 256 ),dtype='float32' ), cmap=plt.cm.gray ) # Tried a different format, also didn't work
        self.im_fig.show()
        self.im_im.axes.figure.canvas.draw()

    def im_callback( self, data ):

        cv_im = self.bridge.imgmsg_to_cv( data, "mono8" ) # Convert Image from ROS Message to greyscale CV Image
        im = np.asarray( cv_im ) # Convert from CV image to numpy array
        #im = np.asarray( cv_im, dtype='float32' ) / 256 # Tried a different format, also didn't work
        self.im_data.append( im )

    def run( self ):

        rospy.init_node('camera_viewer', anonymous=True)

        sub_im = rospy.Subscriber("/move_base/NeuroLocalPlannerWrapper/updated_costmap", OccupancyGrid, callback)

        while not rospy.is_shutdown():
            if self.im_data:
                im = self.im_data.popleft()

                #######################################################
                # The following code is supposed to display the image:
                #######################################################

                self.im_im.set_cmap( 'gray' ) # This doesn't seem to do anything
                self.im_im.set_data( im ) # This won't show greyscale images
                #self.im_ax.imshow( im, cmap=plt.cm.gray ) # If I use this, the code runs unbearably slow
                self.im_im.axes.figure.canvas.draw()

def main():

    viewer = CameraViewer( root='navbot' )
    viewer.run()

if __name__ == '__main__':
    main()
