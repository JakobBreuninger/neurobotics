#include <neuro_local_planner_wrapper/neuro_local_planner_wrapper.h>
#include <pluginlib/class_list_macros.h>

// Register this planner as a BaseLocalPlanner plugin
PLUGINLIB_EXPORT_CLASS(neuro_local_planner_wrapper::NeuroLocalPlannerWrapper, nav_core::BaseLocalPlanner)

namespace neuro_local_planner_wrapper
{
    // Constructor
    // --> Part of interface
    NeuroLocalPlannerWrapper::NeuroLocalPlannerWrapper() : initialized_(false),
                                                 blp_loader_("nav_core", "nav_core::BaseLocalPlanner")
    {
	
    }

    // Destructor
    // --> Part of interface
    NeuroLocalPlannerWrapper::~NeuroLocalPlannerWrapper()
    {
        tc_.reset();
    }

    // Initialize the planner
    // --> Part of interface
    // name:                some string, not important
    // tf:                  this will tell the planner the robots location (i think)
    // costmap_ros:         the costmap
    // Return:              nothing
    void NeuroLocalPlannerWrapper::initialize(std::string name, tf::TransformListener* tf,
                                         costmap_2d::Costmap2DROS* costmap_ros)
    {
        // If we are not ininialized do so
        if (!initialized_)
        {
            // Publishers subscribers
            ros::NodeHandle private_nh("~/" + name);
            g_plan_pub_ = private_nh.advertise<nav_msgs::Path>("global_plan", 1);
            l_plan_pub_ = private_nh.advertise<nav_msgs::Path>("local_plan", 1);
            updated_costmap_pub_ = private_nh.advertise<nav_msgs::OccupancyGrid>("updated_costmap", 1);
            costmap_sub_ = private_nh.subscribe("/move_base/local_costmap/costmap", 1000,
                                                &NeuroLocalPlannerWrapper::updateCostmap, this);
            state_pub_ = private_nh.advertise<std_msgs::Bool>("new_round", 1);

            // Setup tf
            tf_ = tf;

            // Setup the costmap_ros interface
            costmap_ros_ = costmap_ros;
            costmap_ros_->getRobotPose(current_pose_);

            // Get the actual costmap object
            costmap_ = costmap_ros_->getCostmap();

            // Should we use the dwa planner?
            existing_plugin_ = true;
            std::string local_planner = "dwa_local_planner/DWAPlannerROS";

            // If we want to, lets load a local planner plugin to do the work for us
            if (existing_plugin_)
            {
                try
                {
                    tc_ = blp_loader_.createInstance(local_planner);
                    ROS_INFO("Created local_planner %s", local_planner.c_str());
                    tc_->initialize(blp_loader_.getName(local_planner), tf, costmap_ros);
                }
                catch (const pluginlib::PluginlibException& ex)
                {
                    ROS_FATAL("Failed to create plugin");
                    exit(1);
                }
            }

            // We are now initialized
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
    bool NeuroLocalPlannerWrapper::setPlan(const std::vector<geometry_msgs::PoseStamped>& orig_global_plan)
    {
        // Check if the planner has been initialized
        if (!initialized_)
        {
            ROS_ERROR("This planner has not been initialized, please call initialize() before using this planner");
            return false;
        }

        // Safe the global plan
        global_plan_.clear();
        global_plan_ = orig_global_plan;

        // Set the goal position so we can check if we have arrived or not
        goal_.position.x = orig_global_plan.at(orig_global_plan.size() - 1).pose.position.x;
        goal_.position.y = orig_global_plan.at(orig_global_plan.size() - 1).pose.position.y;

        // If we use the dwa:
        // This code is copied from the dwa_planner
        if (existing_plugin_)
        {
            if (!tc_->setPlan(orig_global_plan))
            {
                ROS_ERROR("Failed to set plan for existing plugin");
                return false;
            }
        }
        return true;
    }

    // Compute the velocity commands
    // --> Part of interface
    // cmd_vel:             fill this vector with our velocity commands (the actual output we're producing)
    // Return:              True if we didn't fail
    bool NeuroLocalPlannerWrapper::computeVelocityCommands(geometry_msgs::Twist& cmd_vel)
    {
        // Should we use the network as a planner or the dwa planner?
        if (!existing_plugin_)
        {
            // Lets drive in circles
            cmd_vel.angular.z = 0.0;
            cmd_vel.linear.x = 0.5;
            return true;
        }
        // Use the existing local planner plugin
        else
        {
            geometry_msgs::Twist cmd;

            if(tc_->computeVelocityCommands(cmd))
            {
                cmd_vel = cmd;
                return true;
            }
            else
            {
                ROS_ERROR("Failed computing a command");
                return false;
            }
        }
    }


    // Tell if goal was reached
    // --> Part of interface
    // Return:              True if goal pose was reached
    bool NeuroLocalPlannerWrapper::isGoalReached()
    {
        // Get current position
        costmap_ros_->getRobotPose(current_pose_);

        // Get distance from position to goal, probably there is a better way to do this
        double dist = sqrt(pow((current_pose_.getOrigin().getX() - goal_.position.x
                                + costmap_->getSizeInMetersX()/2), 2.0)
                           + pow((current_pose_.getOrigin().getY()  - goal_.position.y
                                  + costmap_->getSizeInMetersY()/2), 2.0));

        // More or less an arbitrary number. With above dist calculation this seems to be te best the robot can do...
        if(dist < 0.2)
        {
            ROS_INFO("We made it to the goal!");

            // Publish that a new round can be started with the stage_sim_bot
            std_msgs::Bool new_round;
            new_round.data = true;
            state_pub_.publish(new_round);
            global_plan_.clear();
            return true;
        }
        else
        {
            return false;
        }
    }


    // Callback function for the subscriber to the local costmap
    // costmap:             this is the costmap message
    // Return:              nothing
    void NeuroLocalPlannerWrapper::updateCostmap(nav_msgs::OccupancyGrid costmap)
    {
        updated_costmap_ = costmap;

        // Get costmap size
        int width = updated_costmap_.info.width;
        int height = updated_costmap_.info.height;

        // Change the costmap
        for (int i = 0; i < height; i++)
        {
            for (int j = 0; j < width; j++)
            {
                if (updated_costmap_.data[i * width + j] < 100)
                {
                    updated_costmap_.data[i * width + j] = 50;
                }
            }
        }

        // Transform the global plan into costmap coordinates
        unsigned int c_x, c_y;
        double x, y;
        for (unsigned int i = 0; i < global_plan_.size(); i++)
        {
            // Get world coordinates of the current global path point
            x = global_plan_.at(i).pose.position.x - costmap_->getSizeInMetersX()/2;
            y = global_plan_.at(i).pose.position.y - costmap_->getSizeInMetersY()/2;

            // Transform to costmap coordinates of the current global path point if possible
            if (costmap_->worldToMap(x, y, c_x, c_y))
            {
                updated_costmap_.data[c_x + c_y * width] = 0;
            }
        }

        updated_costmap_pub_.publish(updated_costmap_);
    }
};
