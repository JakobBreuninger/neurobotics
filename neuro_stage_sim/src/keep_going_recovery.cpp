#include <keep_going_recovery/keep_going_recovery.h>
#include <pluginlib/class_list_macros.h>

//register this planner as a RecoveryBehavior plugin
PLUGINLIB_DECLARE_CLASS(keep_going_recovery, KeepGoingRecovery, keep_going_recovery::KeepGoingRecovery,
                        nav_core::RecoveryBehavior)

namespace keep_going_recovery
{
    KeepGoingRecovery::KeepGoingRecovery(): initialized_(false){}

    void KeepGoingRecovery::initialize(std::string name, tf::TransformListener* tf,
                                    costmap_2d::Costmap2DROS* global_costmap, costmap_2d::Costmap2DROS* local_costmap)
    {
        if(!initialized_)
        {
            name_ = name;

            ros::NodeHandle private_nh("~/" + name_);

            state_pub_ = private_nh.advertise<std_msgs::Bool>("new_round", 1);

            initialized_ = true;
        }
        else
        {
            ROS_ERROR("You should not call initialize twice on this object, doing nothing");
        }
    }

    KeepGoingRecovery::~KeepGoingRecovery()
    {

    }

    void KeepGoingRecovery::runBehavior()
    {
        if(!initialized_){
            ROS_ERROR("This object must be initialized before runBehavior is called");
            return;
        }

        ROS_ERROR("We're stuck! Lets beam to a new pose with new goal!");

        // Publish that a new round has to be started with the stage_sim_bot
        std_msgs::Bool new_round;
        new_round.data = true;
        state_pub_.publish(new_round);
    }
};
