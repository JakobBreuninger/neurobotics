#!/usr/bin/env python

import rospy
from ros_handler import ROSHandler
from ddpg import DDPG


# Hyper parameters
ONLINE = True
MIN_BUFFER_SIZE = 500


def main():

    # Initialize the ANNs
    agent = DDPG()

    rospy.init_node("neuro_deep_planner", anonymous=False)

    ros_handler = ROSHandler(ONLINE)
    ros_handler.on_policy = False

    while not rospy.is_shutdown():

        # If we have a new msg we might have to execute an action and need to put the new experience in the buffer
        if ros_handler.new_msg():

            # If we are on policy we need to create the new action with the actor net
            if ONLINE:
                ros_handler.new_action = agent.get_action(ros_handler.new_state_temp)

            # Send back the action to execute
            ros_handler.publish_action()

            # Safe the past state and action + the reward and new state into the replay buffer
            agent.set_experience(ros_handler.old_state, ros_handler.old_action, 0.0,
                                 ros_handler.new_state, False)

        #elif agent.get_buffer_size() > MIN_BUFFER_SIZE:

            # Train the network!
            # agent.train()


if __name__ == '__main__':
    main()
