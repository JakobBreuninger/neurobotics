#include "ros/ros.h"
#include "std_msgs/Bool.h"

void botCallback(const std_msgs::Bool new_round)
{
    ROS_ERROR("Whoop!");
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "neuro_stage_sim_bot");

    ros::NodeHandle n;

    ROS_ERROR("You can hear me");

    // Listener for the messages coming from move_base
    ros::Subscriber sub = n.subscribe("neuro_stage_sim_bot/new_round", 1000, botCallback);

    // Publisher for the messages going to move_base
    //ros::Publisher pub

    ros::spin();

    return 0;
}