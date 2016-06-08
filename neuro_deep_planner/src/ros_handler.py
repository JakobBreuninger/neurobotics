#!/usr/bin/env python


import rospy
import numpy as np
from neuro_local_planner_wrapper.msg import Transition
from geometry_msgs.msg import Twist, Vector3


class ROSHandler:

    def __init__(self):

        # Initially assumed Input size, since init is false these values will be updated with the first received msg
        self.__init = False
        self.depth = 4
        self.height = 84
        self.width = 84

        self.state = np.zeros((self.width, self.height, self.depth), dtype='float')

        self.reward = 0.0

        self.__sub = rospy.Subscriber("/move_base/NeuroLocalPlannerWrapper/transition", Transition,
                                      self.input_callback)
        self.__pub = rospy.Publisher("action", Twist, queue_size=10)

        self.__new_msg_flag = False

    def input_callback(self, transition_msg):

        # If msg is received for the first time adjust parameters

        if not self.__init:
            self.depth = transition_msg.depth
            self.width = transition_msg.width
            self.height = transition_msg.height
            self.state = np.zeros((self.depth, self.width, self.height), dtype='float')
            self.__init = True

        # Lets update the new costmap its possible that we need to switch some axes here...
        temp_state = np.asarray(transition_msg.state_representation).reshape(self.depth, self.height, self.width).\
            swapaxes(1, 2)
        self.state = np.rollaxis(temp_state, 0, 3)

        # Lets update the new reward
        self.reward = transition_msg.reward

        # Check if episode is done or not
        self.is_episode_finished = transition_msg.is_episode_finished

        # We have received a new msg
        self.__new_msg_flag = True

    def publish_action(self, action):

        # Generate msg output
        vel_cmd = Twist(Vector3(action[0],0,0),Vector3(0,0,action[1]))

        # Send the action back
        self.__pub.publish(vel_cmd)

    def new_msg(self):

        # Return true if new msg arrived only once for every new msg
        output = False
        if self.__new_msg_flag:
            output = True
            self.__new_msg_flag = False

        return output
