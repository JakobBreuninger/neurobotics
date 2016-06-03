#!/usr/bin/env python

import rospy
from state_visualizer import CostmapVisualizer


def main():

    rospy.init_node("neuro_input_visualizer", anonymous=False)
    viewer = CostmapVisualizer()
    while not rospy.is_shutdown():
        viewer.run()


if __name__ == '__main__':
    main()
