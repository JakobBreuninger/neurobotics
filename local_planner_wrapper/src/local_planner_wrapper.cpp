#include <local_planner_wrapper/local_planner_wrapper.h>
#include <pluginlib/class_list_macros.h>

// Register this planner as a BaseLocalPlanner plugin
PLUGINLIB_EXPORT_CLASS(local_planner_wrapper::LocalPlannerWrapper, nav_core::BaseLocalPlanner)

namespace local_planner_wrapper
{
    // Constructor
    // --> Part of interface
    LocalPlannerWrapper::LocalPlannerWrapper() : initialized_(false),
                                                 blp_loader_("nav_core", "nav_core::BaseLocalPlanner")
    {

    }

    // Desctructor
    // --> Part of interface
    LocalPlannerWrapper::~LocalPlannerWrapper()
    {
        tc_.reset();
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
            updated_costmap_pub_ = private_nh.advertise<nav_msgs::OccupancyGrid>("updated_costmap", 1);
            costmap_sub_ = private_nh.subscribe("/move_base/local_costmap/costmap", 1000,
                                                &LocalPlannerWrapper::filterCostmap, this);
            costmap_update_sub_ = private_nh.subscribe("/move_base/local_costmap/costmap_updates", 1000,
                                                &LocalPlannerWrapper::updateCostmap, this);

            // --- Just for testing: ---
            // initialization of cost map as only updates are received
            filtereded_costmap_ = nav_msgs::OccupancyGrid();

            filtereded_costmap_.info.height = 80;
            filtereded_costmap_.info.width = 80;
            filtereded_costmap_.info.resolution = 0.05;
            filtereded_costmap_.info.origin.position.x = -1.95;
            filtereded_costmap_.info.origin.position.y = -1.95;
            filtereded_costmap_.info.origin.position.z = 0.0;
            filtereded_costmap_.info.origin.orientation.x = 0.0;
            filtereded_costmap_.info.origin.orientation.y = 0.0;
            filtereded_costmap_.info.origin.orientation.z = 0.0;
            filtereded_costmap_.info.origin.orientation.w = 1.0;

            std::vector<int8_t> data(6400,1);
            filtereded_costmap_.data = data;

            // -------------------------------------


            // Setup tf
            tf_ = tf;

            // Setup the costmap_ros interface
            costmap_ros_ = costmap_ros;
            costmap_ros_->getRobotPose(current_pose_);

            // Get the actual costmap object
            costmap_ = costmap_ros_->getCostmap();

            // Should we use the dwa planner?
            dwa_ = true;
            std::string local_planner = "base_local_planner/TrajectoryPlannerROS";

            // If we want to, lets load a local planner plugin to do the work for us
            if (dwa_)
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
    bool LocalPlannerWrapper::setPlan(const std::vector<geometry_msgs::PoseStamped>& orig_global_plan)
    {
        // Check if the planner has been initialized
        if (!initialized_)
        {
            ROS_ERROR("This planner has not been initialized, please call initialize() before using this planner");
            return false;
        }

        // Safe the global plan
        global_plan_ = orig_global_plan;

        // If we use the dwa:
        // This code is copied from the dwa_planner
        if (dwa_)
        {
            if(tc_->setPlan(orig_global_plan))
            {
                ROS_ERROR("Successfully set plan!!!");
            }
        }
        return true;
    }

    // Compute the velocity commands
    // --> Part of interface
    // cmd_vel:             fill this vector with our velocity commands (the actual output we're producing)
    // Return:              True if we didn't fail
    bool LocalPlannerWrapper::computeVelocityCommands(geometry_msgs::Twist& cmd_vel)
    {
        // Should we use the network as a planner or the dwa planner?
        if (!dwa_)
        {
            // Lets drive in circles
            cmd_vel.angular.z = 0.1;
            cmd_vel.linear.x = 0.1;
            return true;
        }
        // This code is copied from the dwa_planner source code...
        else
        {
            geometry_msgs::Twist cmd;

            if(tc_->computeVelocityCommands(cmd))
            {
                // ROS_ERROR("Successfully computed a command");
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
    bool LocalPlannerWrapper::isGoalReached()
    {
        if(dwa_)
        {
            return tc_->isGoalReached();
        }
        else
        {
            return false;
        }
    }


    // Callback function for the subscriber to the local costmap update
    // costmap_update:      this is the costmap message
    // Return:              nothing
    void LocalPlannerWrapper::updateCostmap(map_msgs::OccupancyGridUpdate costmap_update) {

        std::cout << "Costmap update received -> update costmap!!!" << std::endl;

        int index = 0;

        for(int y = costmap_update.y; y < costmap_update.y + costmap_update.height; y++)
        {
            for(int x = costmap_update.x; x < costmap_update.x + costmap_update.width; x++)
            {
                filtereded_costmap_.data[getIndex(x,y)] = costmap_update.data[index++];
            }
        }
        filterCostmap(filtereded_costmap_);

    }


    // Get index for costmap update
    // x:
    // y:
    // Return:
    int LocalPlannerWrapper::getIndex(int x, int y)
    {
        int costmap_width = filtereded_costmap_.info.width;
        return y * costmap_width + x;
    }


    // Callback function for the subscriber to the local costmap
    // costmap:             this is the costmap message
    // Return:              nothing
    void LocalPlannerWrapper::filterCostmap(nav_msgs::OccupancyGrid costmap)
    {
        filtereded_costmap_ = costmap;

        // Get costmap size
        int width = filtereded_costmap_.info.width;
        int height = filtereded_costmap_.info.height;

        // Change the costmap
        for (int i = 0; i < height; i++)
        {
            for (int j = 0; j < width; j++)
            {
                if (filtereded_costmap_.data[i * width + j] < 99)
                {
                    filtereded_costmap_.data[i * width + j] = 50;
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
                //ROS_ERROR("X: %f, y: %f, i: %i\n", x, y, i);
                filtereded_costmap_.data[c_x + c_y*width] = 0;
            }


        }

        std::cout << "And now publish filtered costmap on specified topic!!!" << std::endl;

        updated_costmap_pub_.publish(filtereded_costmap_);
    }

};
