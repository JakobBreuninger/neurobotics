#ifndef LOCAL_PLANNER_WRAPPER_LOCAL_PLANNER_WRAPPER_H_
#define LOCAL_PLANNER_WRAPPER_LOCAL_PLANNER_WRAPPER_H_

#include <tf/transform_listener.h>
#include <angles/angles.h>
#include <nav_msgs/Odometry.h>
#include <costmap_2d/costmap_2d_ros.h>
#include <nav_core/base_local_planner.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Twist.h>

// We use namespaces to keep things seperate under all the planners
namespace local_planner_wrapper
{
    class LocalPlannerWrapper : public nav_core::BaseLocalPlanner
    {
        public:

            // Constructor
            // --> Part of interface
            LocalPlannerWrapper();

            // Desctructor
            // --> Part of interface
            ~LocalPlannerWrapper();

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

            // Publish the local plan
            // path:                could be extrapolated from our velocity commands? probably really short line or
            //                      curve. Maybe interesting for visualisation/debugging
            // Return:              nothing
            void publishLocalPlan(std::vector<geometry_msgs::PoseStamped>& path);

            // Publish the global plan (could be necessary if we don't use the full global plan)
            // path:                this is the part of the global plan we're following at the moment (could be a
            //                      fraction of the full plan)
            // Return:              nothing
            void publishGlobalPlan(std::vector<geometry_msgs::PoseStamped>& path);

            // Listener to get our pose on the map
            tf::TransformListener* tf_;

            // For visualisation, publishers of global and local plan
            ros::Publisher g_plan_pub_, l_plan_pub_;

            // Our costmap
            costmap_2d::Costmap2DROS* costmap_ros_;

            // Our current pose
            tf::Stamped<tf::Pose> current_pose_;

            // Are we initialized?
            bool initialized_;
    };
};
#endif
