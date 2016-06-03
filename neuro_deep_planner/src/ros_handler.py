#!/usr/bin/env python


import rospy
import numpy as np
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Twist


class ROSHandler:

    def __init__(self, on_policy):

        self.on_policy = on_policy

        # Initially assumed Input size, since init is false these values will be updated with the first received msg
        self.init = False
        self.depth = 4
        self.height = 80
        self.width = 80

        self.new_action = np.zeros(2, dtype='float')
        self.old_action = np.zeros(2, dtype='float')

        self.new_state = np.zeros((self.depth, self.width, self.height), dtype='float')
        self.old_state = np.zeros((self.depth, self.width, self.height), dtype='float')

        self.reward = 0.0

        self.sub = rospy.Subscriber("/move_base/NeuroLocalPlannerWrapper/constcutive_costmaps", OccupancyGrid,
                                    self.input_callback)
        self.pub = rospy.Publisher("/Full/Path/Goes/Here", Twist)

        self.new_msg_flag = False

    def input_callback(self, data):

        # If msg is received for the first time adjust parameters
        if not self.init:
            #self.depth = data.info.depth
            self.width = data.info.width
            self.height = data.info.height
            self.new_state = np.zeros((self.depth, self.width, self.height), dtype='float')
            self.new_state = np.zeros((self.depth, self.width, self.height), dtype='float')
            self.init = True

        # Safe the old costmap and action before we update the new one
        self.old_state = self.new_state
        self.old_action = self.new_action

        # Lets update the new costmap its possible that we need to switch some axes here...
        self.new_state = np.asarray(data.data).reshape(4, 80, 80).swapaxes(1, 2)

        # Lets update the new reward
        #self.reward = data.reward

        # Lets update the new action
        #if not self.on_policy:
        #    self.new_action[0] = data.action.linear[0]
        #    self.new_action[1] = data.action.angular[2]

        # We have received a new msg
        self.new_msg_flag = True

    def publish_action(self):

        vel_cmd = Twist()

        vel_cmd.linear[0] = self.new_action[0]
        vel_cmd.linear[1] = 0.0
        vel_cmd.linear[2] = 0.0

        vel_cmd.angular[0] = 0.0
        vel_cmd.angular[1] = 0.0
        vel_cmd.angular[3] = self.new_action[1]

        # Send the action back
        self.pub.publish(self.new_action)

    def new_msg(self):

        output = False
        if self.new_msg_flag:
            output = True
            self.new_msg_flag = False

        return output
