#!/usr/bin/env python

import rospy
import numpy as np

from state_visualizer import CostmapVisualizer
from neuro_local_planner_wrapper.msg import Transition


# Global variable (not ideal but works)
viewer = CostmapVisualizer()

def callback(data):

    if not data.is_episode_finished:

        data_1d = np.asarray([(100 - data) for data in data.state_representation])

        data_3d = data_1d.reshape(4, 84, 84).swapaxes(1, 2)

        data_3d = np.rollaxis(data_3d, 0, 3)

        # Make this a state batch with just one state in the batch
        data_3d = np.expand_dims(data_3d, axis=0)
        viewer.set_data(data_3d)


def main():

    rospy.init_node("neuro_input_visualizer", anonymous=False)

    subscriber = rospy.Subscriber("/move_base/NeuroLocalPlannerWrapper/transition", Transition, callback)
    while not rospy.is_shutdown():
        viewer.run()


if __name__ == '__main__':
    main()
