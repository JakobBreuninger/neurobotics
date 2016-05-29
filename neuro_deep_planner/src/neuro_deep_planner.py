#!/usr/bin/env python

import rospy
import numpy as np
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Twist


class InputSubscriber:


    def __init__(self):

        self.on_policy = False

        self.depth = 4
        self.height = 80
        self.width = 80#

        self.new_action = np.zeros(2, dtype='float')
        self.old_action = np.zeros(2, dtype='float')

        self.new_costmap = np.zeros((self.depth, self.width, self.height), dtype='float')
        self.old_costmap = np.zeros((self.depth, self.width, self.height), dtype='float')

        self.reward = 0.0

        self.init = False
        self.sub = rospy.Subscriber("/move_base/NeuroLocalPlannerWrapper/updated_costmap", OccupancyGrid,
                                    self.input_callback)
        self.pub = rospy.Publisher("/Full/Path/Goes/Here", Twist)

        self.new_msg = False


    def input_callback(self, data):

        # If msg is received for the first time adjust parameters
        if not self.init:
            self.depth = data.info.depth
            self.width = data.info.width
            self.height = data.info.height
            self.new_costmap = np.zeros((self.depth, self.width, self.height), dtype='float')
            self.new_costmap = np.zeros((self.depth, self.width, self.height), dtype='float')
            self.init = True

        # Safe the old costmap and action before we update the new one
        self.old_costmap = self.new_costmap
        self.old_action = self.new_action

        # Lets update the new costmap
        for i in range(self.depth):
            for j in range(self.width):
                for k in range(self.height):
                    self.new_costmap[i][j][k] = data.data[j + self.height * k + self.height * self.width * i]

        # Lets update the new reward
        self.reward = data.reward

        # Lets update the new action
        if not self.on_policy:
            self.new_action[0] = data.action.linear[0]
            self.new_action[1] = data.action.angular[2]

        # We have received a new msg
        self.new_msg = True


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


class ReplayBuffer:

    def __init__(self):


    def set_experience(self, old_costmap, action, reward, new_costmap):


    def get_experience(self):


    def safe_buffer(self):
        # Empty

    def load_buffer(self):
        # Empty



def main():

    # Initialize the ANNs
    actor = Q_net()
    critic = Q_net()

    rospy.init_node("neuro_deep_planner", anonymous=False)

    subscriber = InputSubscriber()
    subscriber.on_policy = False

    replay_buffer = ReplayBuffer()

    while not rospy.is_shutdown():

        # If we have a new msg we might have to execute an action and need to put the new experience in the buffer
        if subscriber.new_msg:

            # If we are on policy we need to create the new action with the actor net
            if subscriber.on_policy:
                subscriber.new_action = actor(subscriber.new_costmap)

            # Safe the past state and action + the reward and new state into the replay buffer
            replay_buffer.set_experience(subscriber.old_costmap, subscriber.old_action, subscriber.reward,
                                         subscriber.new_costmap)

            # Send back the action to execute
            subscriber.publish_action()
            subscriber.new_msg = False
        else:
            # Train the network!


if __name__ == '__main__':
    main()
