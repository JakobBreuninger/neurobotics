#!/usr/bin/env python

import rospy
from ros_handler import ROSHandler
from ddpg import DDPG


# Hyper parameters
ONLINE = True

def main():

    # Initialize the ANNs
    agent = DDPG()

    rospy.init_node("neuro_deep_planner", anonymous=False)

    ros_handler = ROSHandler()
    ros_handler.on_policy = False

    while not rospy.is_shutdown():

        # If we have a new msg we might have to execute an action and need to put the new experience in the buffer
        if ros_handler.new_msg():

            # Send back the action to execute
            ros_handler.publish_action(agent.get_action(ros_handler.state))

            # Safe the past state and action + the reward and new state into the replay buffer
            agent.set_experience(ros_handler.state, ros_handler.reward, ros_handler.is_episode_finished)

        else:
            # Train the network!
            agent.train()


if __name__ == '__main__':
    main()
