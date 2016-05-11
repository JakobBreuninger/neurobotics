#include <local_planner_wrapper/local_planner_wrapper.h>
#include <cmath>
#include <ros/console.h>
#include <pluginlib/class_list_macros.h>
#include <nav_msgs/Path.h>

// Register this planner as a BaseLocalPlanner plugin
PLUGINLIB_EXPORT_CLASS(local_planner_wrapper::LocalPlannerWrapper, nav_core::BaseLocalPlanner)

namespace local_planner_wrapper
{
    // Constructor
    // --> Part of interface
    LocalPlannerWrapper::LocalPlannerWrapper() : initialized_(false)
    {
	
    }

    // Desctructor
    // --> Part of interface
    LocalPlannerWrapper::~LocalPlannerWrapper()
    {

    }

    // Initialize the planner
    // --> Part of interface
    // name:                some string, not important
    // tf:                  this will tell the planner the robots location (i think)
    // costmap_ros:         the costmap
    // Return:              nothing
    void LocalPlannerWrapper::initialize(std::string name, tf::TransformListener* tf,
        costmap_2d::Costmap2DROS* costmap_ros)
    {
        // If we are not ininialized do so
        if (!initialized_)
        {
            // Publishers subscribers
            ros::NodeHandle private_nh("~/" + name);
            g_plan_pub_ = private_nh.advertise<nav_msgs::Path>("global_plan", 1);
            l_plan_pub_ = private_nh.advertise<nav_msgs::Path>("local_plan", 1);
            updated_costmap_pub_ = 
                private_nh.advertise<nav_msgs::OccupancyGrid>("updated_costmap", 1);
            costmap_sub_ = 
                private_nh.subscribe<nav_msgs::OccupancyGrid>("", 1000, updateCostmap);

            // Setup tf
            tf_ = tf;

            // Setup the costmap
            costmap_ros_ = costmap_ros;
            costmap_ros_->getRobotPose(current_pose_);
            costmap_2d::Costmap2D* costmap = costmap_ros_->getCostmap();
            updated_costmap_ = *costmap_ros_;

            // We are now ininialized
            initialized_ = true;
        }
        else
        {
            ROS_WARN("This planner has already been initialized, doing nothing.");
        }
    }

    // Sets the plan
    // --> Part of interface
    // orig_global_plan:    this is the global plan we're supposed to follow (a vector of positions forms the
    //                      line)
    // Return:              True if plan was succesfully received...
    bool LocalPlannerWrapper::setPlan(const std::vector<geometry_msgs::PoseStamped>& orig_global_plan)
    {
        // Check if the planner has been initialized
        if (!initialized_)
        {
            ROS_ERROR("This planner has not been initialized, please call initialize() before using this planner");
            return false;
        }

        ROS_INFO("We have a plan but we're doing nothing with it :D");
        return true;
    }

    // Compute the velocity commands
    // --> Part of interface
    // cmd_vel:             fill this vector with our velocity commands (the actual output we're producing)
    // Return:              True if we didn't fail
    bool LocalPlannerWrapper::computeVelocityCommands(geometry_msgs::Twist& cmd_vel)
    {
        // Lets drive in circles
        cmd_vel.angular.z = 0.1;
        cmd_vel.linear.x = 0.05;

        return true;
    }

    // Tell if goal was reached
    // --> Part of interface
    // Return:              True if goal pose was reached
    bool LocalPlannerWrapper::isGoalReached()
    {
        return false;
    }

    // Publish the local plan
    // path:                could be extrapolated from our velocity commands? probably really short line or
    //                      curve. Maybe interesting for visualisation/debugging
    // Return:              nothing
    void LocalPlannerWrapper::publishLocalPlan(std::vector<geometry_msgs::PoseStamped>& path)
    {
        // base_local_planner::publishPlan(path, l_plan_pub_);
    }

    // Publish the global plan (could be necessary if we don't use the full global plan)
    // path:                this is the part of the global plan we're following at the moment (could be a
    //                      fraction of the full plan)
    // Return:              nothing
    void LocalPlannerWrapper::publishGlobalPlan(std::vector<geometry_msgs::PoseStamped>& path)
    {
        // base_local_planner::publishPlan(path, g_plan_pub_);
    }

    void LocalPlannerWrapper::updateCostmap()
    {
        updated_costmap_.
        updated_costmap_pub_.publish(updated_costmap_);
    }

};
