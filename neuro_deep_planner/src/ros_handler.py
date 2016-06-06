#!/usr/bin/env python


import rospy
import numpy as np
from neuro_local_planner_wrapper.msg import Transition
from geometry_msgs.msg import Twist


class ROSHandler:

    def __init__(self):

        # Initially assumed Input size, since init is false these values will be updated with the first received msg
        self.__init = False
        self.depth = 4
        self.height = 80
        self.width = 80

        self.state = np.zeros((self.depth, self.width, self.height), dtype='float')

        self.reward = 0.0

        self.__sub = rospy.Subscriber("/move_base/NeuroLocalPlannerWrapper/constcutive_costmaps", Transition,
                                      self.input_callback)
        self.__pub = rospy.Publisher("/Full/Path/Goes/Here", Twist)

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
        self.state = np.asarray(transition_msg.state_representation).reshape(4, 80, 80).swapaxes(1, 2)

        # Lets update the new reward
        self.reward = transition_msg.reward

        # We have received a new msg
        self.__new_msg_flag = True

    def publish_action(self, action):

        # Generate msg output
        vel_cmd = Twist()

        vel_cmd.linear[0] = action[0]
        vel_cmd.linear[1] = 0.0
        vel_cmd.linear[2] = 0.0

        vel_cmd.angular[0] = 0.0
        vel_cmd.angular[1] = 0.0
        vel_cmd.angular[3] = action[1]

        # Send the action back
        self.__pub.publish(vel_cmd)

    def new_msg(self):

        # Return true if new msg arrived only once for every new msg
        output = False
        if self.__new_msg_flag:
            output = True
            self.__new_msg_flag = False

        return output
