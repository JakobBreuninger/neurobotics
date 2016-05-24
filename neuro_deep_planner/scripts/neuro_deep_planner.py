#!/usr/bin/env python

import rospy
from nav_msgs.msg import OccupancyGrid
import numpy as np


def callback(data):
    rospy.loginfo(rospy.get_caller_id() + "Success")


def listener():

    rospy.init_node("neuro_deep_planner", anonymous=False)

    rospy.Subscriber("/move_base/NeuroLocalPlannerWrapper/updated_costmap", OccupancyGrid, callback)

    rospy.spin()


if __name__ == '__main__':
    listener()
