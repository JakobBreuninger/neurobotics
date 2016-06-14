#!/usr/bin/env python


import rospy
import numpy as np
from neuro_local_planner_wrapper.msg import Transition
from std_msgs.msg import Bool
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
        self.is_episode_finished = False

        self.__sub_move_base = rospy.Subscriber("/move_base/NeuroLocalPlannerWrapper/transition", Transition,
                                                self.input_callback)
        self.__sub_setting = rospy.Subscriber("/noise_flag", Bool, self.setting_callback)
        self.__pub = rospy.Publisher("neuro_deep_planner/action", Twist, queue_size=10)

        self.__new_msg_flag = False
        self.__new_setting_flag = False
        self.noise_flag = True

    def input_callback(self, transition_msg):

        # If msg is received for the first time adjust parameters

        if not self.__init:
            self.depth = transition_msg.depth
            self.width = transition_msg.width
            self.height = transition_msg.height
            self.state = np.zeros((self.depth, self.width, self.height), dtype='float')
            self.__init = True

        # Lets update the new reward
        self.reward = transition_msg.reward

        # Check if episode is done or not
        self.is_episode_finished = transition_msg.is_episode_finished

        # Lets update the new costmap its possible that we need to switch some axes here...
        if not self.is_episode_finished:
            temp_state = np.asarray(transition_msg.state_representation).reshape(self.depth, self.height, self.width).\
                swapaxes(1, 2)
            self.state = np.rollaxis(temp_state, 0, 3)

            # Normalize!
            self.state = self.state.astype(float)
            self.state = np.divide(self.state, 100.0)

        # We have received a new msg
        self.__new_msg_flag = True

    def setting_callback(self, setting_msg):

        # If msg is received for the first time adjust parameters

        self.noise_flag = setting_msg.data

        # We have received a setting
        self.__new_setting_flag = True

    def publish_action(self, action):

        # Generate msg output
        vel_cmd = Twist(Vector3(action[0], 0, 0), Vector3(0, 0, action[1]))

        # Send the action back
        self.__pub.publish(vel_cmd)

    def new_msg(self):

        # Return true if new msg arrived only once for every new msg
        output = False
        if self.__new_msg_flag:
            output = True
            self.__new_msg_flag = False

        return output

    def new_setting(self):

        # Return true if new msg arrived only once for every new msg
        output = False
        if self.__new_setting_flag:
            output = True
            self.__new_setting_flag = False

        return output

