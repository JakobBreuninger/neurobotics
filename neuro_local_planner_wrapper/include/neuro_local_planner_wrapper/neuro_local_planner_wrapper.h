#ifndef NEURO_LOCAL_PLANNER_WRAPPER_NEURO_LOCAL_PLANNER_WRAPPER_H_
#define NEURO_LOCAL_PLANNER_WRAPPER_NEURO_LOCAL_PLANNER_WRAPPER_H_

#include <tf/transform_listener.h>
#include <angles/angles.h>
#include <nav_msgs/Odometry.h>
#include <costmap_2d/costmap_2d_ros.h>
#include <nav_core/base_local_planner.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/Twist.h>
#include <nav_msgs/Path.h>
#include <std_msgs/Bool.h>
#include <pluginlib/class_loader.h>

#include <neuro_local_planner_wrapper/Transition.h>

#include <nav_msgs/OccupancyGrid.h>
#include <map_msgs/OccupancyGridUpdate.h>
#include <sensor_msgs/LaserScan.h>

#include <visualization_msgs/MarkerArray.h> // to_delete

#include <tf/tf.h>

// We use namespaces to keep things seperate under all the planners
namespace neuro_local_planner_wrapper
{
    class NeuroLocalPlannerWrapper : public nav_core::BaseLocalPlanner
    {
        public:

            // Constructor
            // --> Part of interface
            NeuroLocalPlannerWrapper();

            // Desctructor
            // --> Part of interface
            ~NeuroLocalPlannerWrapper();

            // Initialize the planner
            // --> Part of interface
            // name:                some string, not important
            // tf:                  this will tell the planner the robots location (i think)
            // costmap_ros:         the costmap
            // Return:              nothing
            void initialize(std::string name, tf::TransformListener* tf, costmap_2d::Costmap2DROS* costmap_ros);

            // Sets the plan
            // --> Part of interface
            // orig_global_plan:    this is the global plan we're supposed to follow (a vector of positions forms the
            //                      line)
            // Return:              True if plan was succesfully received...
            bool setPlan(const std::vector<geometry_msgs::PoseStamped>& orig_global_plan);

            // Compute the velocity commands
            // --> Part of interface
            // cmd_vel:             fill this vector with our velocity commands (the actual output we're producing)
            // Return:              True if we didn't fail
            bool computeVelocityCommands(geometry_msgs::Twist& cmd_vel);

            // Tell if goal was reached
            // --> Part of interface
            // Return:              True if goal pose was reached
            bool isGoalReached();

        private:

            // Callback function for the subscriber to laser scan
            // laser_scan:          this is the laser scan message
            // Return:              nothing
            void getLaserScanPoints(sensor_msgs::LaserScan laser_scan);

            void initializeCustomizedCostmap();

            void addMarkerToArray(double x, double y, std::string frame, ros::Time stamp); // to_delete

            // Listener to get our pose on the map
            tf::TransformListener* tf_;


            // --- Publisher & Subscriber ---

            // For visualisation, publishers of global and local plan
            ros::Publisher g_plan_pub_, l_plan_pub_; // TODO: never used right?

            // Publisher to the stage_sim_bot
            ros::Publisher state_pub_;

            // Subscribe to laser scan topic
            ros::Subscriber laser_scan_sub_;

            // For visualisation, publisher for customized costmap
            ros::Publisher customized_costmap_pub_;

            // Publisher for communication with TensorFlow
            ros::Publisher transition_pub_;

            ros::Publisher marker_array_pub_; // to_delete



            // Our costmap ros interface
            costmap_2d::Costmap2DROS* costmap_ros_;

            // Our actual costmap
            costmap_2d::Costmap2D* costmap_;



            // Customized costmap as state representation of the robot base
            nav_msgs::OccupancyGrid customized_costmap_;

            // Indicates whether customized costmap has been initialized
            bool is_customized_costmap_initialized_;


            // Transition message with actual state representation which is four consecutive costmaps stacked together in one vector and actual reward
            neuro_local_planner_wrapper::Transition transition_;


            visualization_msgs::MarkerArray marker_array_; // to_delete



            // Our current pose
            tf::Stamped<tf::Pose> current_pose_;

            // Our goal pose
            geometry_msgs::Pose goal_;

            // The current global plan in normal and costmap coordinates
            std::vector<geometry_msgs::PoseStamped> global_plan_;

            // Should we use an existing planner plugin to gather samples?
            // Then we need all of these variables...
            bool existing_plugin_;
            pluginlib::ClassLoader<nav_core::BaseLocalPlanner> blp_loader_;
            boost::shared_ptr<nav_core::BaseLocalPlanner> tc_;

            // Are we initialized?
            bool initialized_;
    };
};
#endif
