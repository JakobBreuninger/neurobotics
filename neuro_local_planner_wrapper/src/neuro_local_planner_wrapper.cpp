#include <neuro_local_planner_wrapper/neuro_local_planner_wrapper.h>
#include <pluginlib/class_list_macros.h>

// Register this planner as a BaseLocalPlanner plugin
PLUGINLIB_EXPORT_CLASS(neuro_local_planner_wrapper::NeuroLocalPlannerWrapper, nav_core::BaseLocalPlanner)

double goalTolerance = 0.2;


namespace neuro_local_planner_wrapper
{
    // Constructor
    NeuroLocalPlannerWrapper::NeuroLocalPlannerWrapper() : initialized_(false),
                                                           blp_loader_("nav_core", "nav_core::BaseLocalPlanner") {}

    // Destructor
    NeuroLocalPlannerWrapper::~NeuroLocalPlannerWrapper()
    {
        tc_.reset();
    }


    // Initialize the planner
    void NeuroLocalPlannerWrapper::initialize(std::string name, tf::TransformListener* tf,
                                         costmap_2d::Costmap2DROS* costmap_ros)
    {
        // If we are not initialized do so
        if (!initialized_)
        {
            ros::NodeHandle private_nh("~/" + name);

            // TODO: remove
            // debug_marker_pub_ = private_nh.advertise<visualization_msgs::Marker>( "goal_point", 0 );

            // Publishers & Subscribers
            g_plan_pub_ = private_nh.advertise<nav_msgs::Path>("global_plan", 1);
            l_plan_pub_ = private_nh.advertise<nav_msgs::Path>("local_plan", 1);

            state_pub_ = private_nh.advertise<std_msgs::Bool>("new_round", 1);

            laser_scan_sub_ = private_nh.subscribe("/scan", 1000, &NeuroLocalPlannerWrapper::buildStateRepresentation,
                                                   this);

            customized_costmap_pub_ = private_nh.advertise<nav_msgs::OccupancyGrid>("customized_costmap", 1);

            transition_msg_pub_ = private_nh.advertise<neuro_local_planner_wrapper::Transition>("transition", 1);

            action_pub_ = private_nh.advertise<geometry_msgs::Twist>("action", 1);

            noise_flag_pub_ = private_nh.advertise<std_msgs::Bool>("/noise_flag", 1);

            action_sub_ = private_nh.subscribe("/neuro_deep_planner/action", 1000,
                                               &NeuroLocalPlannerWrapper::callbackAction, this);

            // Setup tf
            tf_ = tf;

            // Setup the costmap_ros interface
            costmap_ros_ = costmap_ros;
            costmap_ros_->getRobotPose(current_pose_);

            // Get the actual costmap object
            costmap_ = costmap_ros_->getCostmap();

            // Initialize customized costmap and transition message
            initializeCustomizedCostmap();
            initializeTransitionMsg();

            // initialize action to zero until first velocity command is computed
            action_ = geometry_msgs::Twist();
            setZeroAction();

            // Should we use the dwa planner?
            existing_plugin_ = false;
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

            is_running_ = false;

            goal_counter_ = 0;
            crash_counter_ = 0;

            file_counter = 0;

            // For plotting
            noise_flag_ = true;
            temp_time_ = (int)ros::Time::now().toSec();
            temp_crash_count_ = 0;
            temp_goal_count_ = 0;

            // To close up too long episodes
            max_time_ = 60;

            // We are now initialized
            initialized_ = true;
        }
        else
        {
            ROS_WARN("This planner has already been initialized, doing nothing.");
        }
    }


    // Sets the plan
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

        is_running_ = true;

        return true;
    }


    // Compute the velocity commands
    bool NeuroLocalPlannerWrapper::computeVelocityCommands(geometry_msgs::Twist& cmd_vel)
    {
        return true;
    }


    // Tell if goal was reached
    bool NeuroLocalPlannerWrapper::isGoalReached()
    {
        return false;
    }


    // Helper function to initialize the state representation
    void NeuroLocalPlannerWrapper::initializeCustomizedCostmap()
    {
        customized_costmap_ = nav_msgs::OccupancyGrid();

        // header
        customized_costmap_.header.frame_id = "/base_footprint";
        customized_costmap_.header.stamp = ros::Time::now();
        customized_costmap_.header.seq = 0;

        // info
        customized_costmap_.info.width = costmap_->getSizeInCellsX(); // e.g. 80
        customized_costmap_.info.height = costmap_->getSizeInCellsY(); // e.g. 80
        customized_costmap_.info.resolution = (float)costmap_->getResolution(); // e.g. 0.05
        customized_costmap_.info.origin.position.x = -costmap_->getSizeInMetersX()/2.0; // e.g.-1.95
        customized_costmap_.info.origin.position.y = -costmap_->getSizeInMetersY()/2.0; // e.g.-1.95
        customized_costmap_.info.origin.position.z = 0.01; // looks better in simulation
        customized_costmap_.info.origin.orientation.x = 0.0;
        customized_costmap_.info.origin.orientation.y = 0.0;
        customized_costmap_.info.origin.orientation.z = 0.0;
        customized_costmap_.info.origin.orientation.w = 1.0;
    }


    // Helper function to initialize the transition message for the planning node
    void NeuroLocalPlannerWrapper::initializeTransitionMsg()
    {
        // header
        transition_msg_.header.frame_id = customized_costmap_.header.frame_id;
        transition_msg_.header.stamp = customized_costmap_.header.stamp;
        transition_msg_.header.seq = 0;

        // info
        transition_msg_.width = customized_costmap_.info.width;
        transition_msg_.height = customized_costmap_.info.height;
        transition_msg_.depth = 4; // use four consecutive maps for state representation 
    }


    // Is called during construction and before the robot is beamed to a new place
    void NeuroLocalPlannerWrapper::setZeroAction()
    {
        action_.linear.x = 0.0;
        action_.linear.y = 0.0;
        action_.linear.z = 0.0;
        action_.angular.x = 0.0;
        action_.angular.y = 0.0;
        action_.angular.z = 0.0;

        action_pub_.publish(action_);
    }


    // Checks if the robot is in collision or not
    bool NeuroLocalPlannerWrapper::isCrashed(double& reward)
    {
        // Get current position of robot
        costmap_ros_->getRobotPose(current_pose_); // in frame odom

        // Compute map coordinates
        int robot_x;
        int robot_y;
        costmap_->worldToMapNoBounds(current_pose_.getOrigin().getX(), current_pose_.getOrigin().getY(), robot_x,
                                     robot_y);

        // This causes a crash not just a critical positions but a little bit before the wall
        // TODO: could be solved nicer by using a different inscribed radius, then: >= 253
        if(costmap_->getCost((unsigned int)robot_x, (unsigned int)robot_y) >= 170)
        {
            crash_counter_++;
            ROS_INFO("We crashed: %d", crash_counter_);
            reward = -1.0;
            return true;
        }
        else
        {
            return false;
        }
    }


    // Checks if the robot reached the goal
    bool NeuroLocalPlannerWrapper::isAtGoal(double& reward)
    {
        // Get current position of robot in odom frame
        costmap_ros_->getRobotPose(current_pose_);

        // Get goal position
        geometry_msgs::PoseStamped goal_position = global_plan_.back();

        // Transform current position of robot to map frame
        tf::StampedTransform stamped_transform;
        try
        {
            // ros::Time(0) gives us the latest available transform
            tf_->lookupTransform(goal_position.header.frame_id, current_pose_.frame_id_, ros::Time(0),
                                 stamped_transform);
        }
        catch (tf::TransformException ex)
        {
            ROS_ERROR("%s",ex.what());
        }

        // Translation TODO: WHYYY? + -> -
        double x_current_pose_map_frame = current_pose_.getOrigin().getX() - stamped_transform.getOrigin().getX();
        double y_current_pose_map_frame = current_pose_.getOrigin().getY() - stamped_transform.getOrigin().getY();

        // Rotation
        double roll, pitch, yaw;
        stamped_transform.getBasis().getRPY(roll, pitch, yaw);
        double x_temp = x_current_pose_map_frame;
        double y_temp = y_current_pose_map_frame;
        x_current_pose_map_frame = cos(yaw)*x_temp - sin(yaw)*y_temp;
        y_current_pose_map_frame = sin(yaw)*x_temp + cos(yaw)*y_temp;

        // Get distance from robot to goal -> for now we only consider distance but I think we could also include
        // orientation
        double dist = sqrt(pow((x_current_pose_map_frame - goal_position.pose.position.x), 2.0)
                           + pow((y_current_pose_map_frame - goal_position.pose.position.y), 2.0));

        // TODO: Publish a marker for visualization of the two dynamic obstacles
//        visualization_msgs::Marker marker;
//
//        marker.header.frame_id = "map";
//        marker.header.stamp = ros::Time();
//        marker.ns = "my_namespace";
//        marker.id = 0;
//        marker.type = visualization_msgs::Marker::CUBE;
//        marker.action = visualization_msgs::Marker::ADD;
//        marker.pose.position.x = x_current_pose_map_frame;
//        marker.pose.position.y = y_current_pose_map_frame;
//        marker.pose.position.z = 0.125;
//        tf::Quaternion quaternion;
//        quaternion.setRPY(0.0, 0.0, 0.0);
//        marker.pose.orientation.x = quaternion.getX();
//        marker.pose.orientation.y = quaternion.getY();
//        marker.pose.orientation.z = quaternion.getZ();
//        marker.pose.orientation.w = quaternion.getW();
//        marker.scale.x = 0.22;
//        marker.scale.y = 0.22;
//        marker.scale.z = 0.5;
//        marker.color.a = 1.0; // Don't forget to set the alpha!
//        marker.color.r = 0.15;
//        marker.color.g = 0.15;
//        marker.color.b = 0.15;
//
//        debug_marker_pub_.publish(marker);

        // Check if the robot has reached the goal
        if(dist < goalTolerance)
        {
            goal_counter_++;
            ROS_INFO("We reached the goal: %d", goal_counter_);
            reward = 1.0;
            return true;
        }
        else
        {
            return false;
        }
    }


    // Publishes the action which is executed by the robot
    void NeuroLocalPlannerWrapper::callbackAction(geometry_msgs::Twist action)
    {
        // Should we use the network as a planner or the dwa planner?
        if (!existing_plugin_)
        {
            // Get action from net
            action_ = action;
        }
        else
        {
            // Use the existing local planner plugin
            geometry_msgs::Twist cmd;
            if(tc_->computeVelocityCommands(cmd))
            {
                if (is_running_) {
                    action_ = cmd;
                }
            }
            else
            {
                ROS_ERROR("Plugin failed computing a command");
            }
        }

        // Publish
        action_pub_.publish(action_);
    }


    // Callback function for the subscriber to the laser scan
    void NeuroLocalPlannerWrapper::buildStateRepresentation(sensor_msgs::LaserScan laser_scan)
    {
        // Safe the
        int now = (int)ros::Time::now().toSec();
        if (noise_flag_ && (now - temp_time_) > 3000)
        {
            temp_crash_count_ = crash_counter_;
            temp_goal_count_ = goal_counter_;

            noise_flag_ = false;

            std_msgs::Bool msg;
            msg.data = 0;

            noise_flag_pub_.publish(msg);

            temp_time_ = now;
        }
        if (!noise_flag_ && (now - temp_time_) > 600)
        {
            std::pair<int, int> temp_count;
            temp_count.first = crash_counter_ - temp_crash_count_;
            temp_count.second = goal_counter_ - temp_goal_count_;

            plot_list_.push_back(temp_count);

            // open file for printing
            std::ofstream outfile;
            std::string my_file_path = "/home/breuning/results/counters.csv";
            outfile.open(my_file_path.c_str());

            for (unsigned int i = 0; i < plot_list_.size(); i++)
            {
                outfile << plot_list_.at(i).first << "," << plot_list_.at(i).second << std::endl;
            }

            outfile.close();

            noise_flag_ = true;

            std_msgs::Bool msg;
            msg.data = 1;

            noise_flag_pub_.publish(msg);

            temp_time_ = now;
        }
        if (is_running_)
        {
            double reward = 0.0;

            if (isCrashed(reward) || isAtGoal(reward))
            {
                // New episode so restart the time count
                start_time_ = ros::Time::now().toSec();

                // This is the last transition published in this episode
                is_running_ = false;

                // Stop moving
                setZeroAction();

                // Publish that a new round can be started with the stage_sim_bot
                std_msgs::Bool new_round;
                new_round.data = 1;
                state_pub_.publish(new_round);

                // Create transition message with empty state
                transition_msg_.header.stamp = laser_scan.header.stamp;
                transition_msg_.header.frame_id = customized_costmap_.header.frame_id;
                transition_msg_.is_episode_finished = 1;
                transition_msg_.reward = reward;

                // clear buffer to get empty state representation
                transition_msg_.state_representation.clear();

                // Publish it
                transition_msg_pub_.publish(transition_msg_);

                // increment seq for next costmap
                transition_msg_.header.seq = transition_msg_.header.seq + 1;
            }
            else if (ros::Time::now().toSec() - start_time_ > max_time_)
            {
                // New episode so restart the time count
                start_time_ = ros::Time::now().toSec();

                // This is the last transition published in this episode
                is_running_ = false;

                // Stop moving
                setZeroAction();

                // Publish that a new round can be started with the stage_sim_bot
                std_msgs::Bool new_round;
                new_round.data = 1;
                state_pub_.publish(new_round);
            }
            else
            {
                // clear costmap/set all pixel gray
                std::vector<int8_t> data(customized_costmap_.info.width*customized_costmap_.info.height,50);
                customized_costmap_.data = data;

                // to_delete: ------
                customized_costmap_.header.stamp = laser_scan.header.stamp;

                // add global plan as white pixel with some gradient to indicate its direction
                addGlobalPlan();

                // add laser scan points as invalid/black pixel
                addLaserScanPoints(laser_scan);

                // publish customized costmap for visualization
                customized_costmap_pub_.publish(customized_costmap_);

                // increment seq for next costmap
                customized_costmap_.header.seq = customized_costmap_.header.seq + 1;

                // build transition message/add actual costmap to buffer
                transition_msg_.state_representation.insert(transition_msg_.state_representation.end(),
                                                            customized_costmap_.data.begin(),
                                                            customized_costmap_.data.end());

                // publish transition message after four consecutive costmaps are available
                if (transition_msg_.state_representation.size() == transition_msg_.width*
                                                                transition_msg_.height*
                                                                transition_msg_.depth)
                {
                    // publish
                    transition_msg_.header.stamp = customized_costmap_.header.stamp;
                    transition_msg_.header.frame_id = customized_costmap_.header.frame_id;
                    transition_msg_.is_episode_finished = 0;
                    transition_msg_.reward = reward;

                    transition_msg_pub_.publish(transition_msg_);

                    // increment seq for next costmap
                    transition_msg_.header.seq = transition_msg_.header.seq + 1;

                    // clear buffer
                    transition_msg_.state_representation.clear();
                }
            }
        }
    }


    // Helper function to generate the transition msg
    void NeuroLocalPlannerWrapper::addLaserScanPoints(const sensor_msgs::LaserScan& laser_scan)
    {
        // get source frame and target frame of laser scan points
        std::string laser_scan_source_frame = laser_scan.header.frame_id;
        std::string laser_scan_target_frame = customized_costmap_.header.frame_id;

        // stamp of first laser point in range
        ros::Time laser_scan_stamp = laser_scan.header.stamp;
        ros::Time customized_costmap_stamp = laser_scan_stamp;

        // update stamp of costmap
        customized_costmap_.header.stamp = customized_costmap_stamp;

        // get transformation between robot base frame and frame of laser scan
        tf::StampedTransform stamped_transform;
        try
        {
            // ros::Time(0) gives us the latest available transform
            tf_->lookupTransform(laser_scan_target_frame, laser_scan_source_frame, ros::Time(0), stamped_transform);
        }
        catch (tf::TransformException ex)
        {
            ROS_ERROR("%s",ex.what());
        }

        // x and y position of laser scan point in frame of laser scan
        double x_position_laser_scan_frame;
        double y_position_laser_scan_frame;

        // x and y position of laser scan point in robot base frame
        double x_position_robot_base_frame;
        double y_position_robot_base_frame;

        // iteration over all laser scan points
        for(unsigned int i = 0; i < laser_scan.ranges.size(); i++)
        {
            if ((laser_scan.ranges.at(i) > laser_scan.range_min) && (laser_scan.ranges.at(i) < laser_scan.range_max))
            {
                // get x and y coordinates of laser scan point in frame of laser scan, z coordinate is ignored as we
                // are working with a 2D costmap
                x_position_laser_scan_frame = laser_scan.ranges.at(i) * cos(laser_scan.angle_min
                                                                            + i * laser_scan.angle_increment);
                y_position_laser_scan_frame = laser_scan.ranges.at(i) * sin(laser_scan.angle_min
                                                                            + i * laser_scan.angle_increment);

                // translation
                x_position_robot_base_frame = x_position_laser_scan_frame + stamped_transform.getOrigin().getX();
                y_position_robot_base_frame = y_position_laser_scan_frame + stamped_transform.getOrigin().getY();

                // rotation
                double roll, pitch, yaw;
                stamped_transform.getBasis().getRPY(roll, pitch, yaw);
                double x_temp = x_position_robot_base_frame;
                double y_temp = y_position_robot_base_frame;
                x_position_robot_base_frame = cos(yaw)*x_temp - sin(yaw)*y_temp;
                y_position_robot_base_frame = sin(yaw)*x_temp + cos(yaw)*y_temp;

                // transformation to costmap coordinates
                int x, y;
                x = (int)round(((x_position_robot_base_frame - customized_costmap_.info.origin.position.x)
                                / costmap_->getSizeInMetersX())*customized_costmap_.info.width-0.5);
                y = (int)round(((y_position_robot_base_frame - customized_costmap_.info.origin.position.y)
                                / costmap_->getSizeInMetersY())*customized_costmap_.info.height-0.5);


                if ((x >=0) && (y >=0) && (x < customized_costmap_.info.width) && (y < customized_costmap_.info.height))
                {
                    customized_costmap_.data[x + y*customized_costmap_.info.width] = 100;
                }
            }
        }
    }

    void NeuroLocalPlannerWrapper::addGlobalPlan()
    {
        // Transform the global plan into costmap coordinates
        // pose given in fixed frame of global plan which is by default "map"
        geometry_msgs::PoseStamped pose_fixed_frame;

        // pose given in global frame of the local cost map
        geometry_msgs::PoseStamped pose_robot_base_frame;

        std::vector<geometry_msgs::Point> global_plan_map_coordinates;
        geometry_msgs::Point a_global_plan_map_coordinate;

        std::vector<geometry_msgs::PoseStamped> global_plan_temp = global_plan_;

        for(std::vector<geometry_msgs::PoseStamped>::iterator it = global_plan_temp.begin(); it != global_plan_temp.end(); it++) {

            // Transform pose from fixed frame of global plan to global frame of local cost map
            pose_fixed_frame = *it;
            try
            {
                pose_fixed_frame.header.stamp = customized_costmap_.header.stamp;
                tf_->waitForTransform(customized_costmap_.header.frame_id, pose_fixed_frame.header.frame_id,
                                      customized_costmap_.header.stamp, ros::Duration(0.2));
                tf_->transformPose(customized_costmap_.header.frame_id, pose_fixed_frame, pose_robot_base_frame);
            }
            catch (tf::TransformException ex)
            {
                ROS_ERROR("%s", ex.what());
            }

            // transformation to costmap coordinates
            int x, y;
            x = (int)round(((pose_robot_base_frame.pose.position.x - customized_costmap_.info.origin.position.x)
                            / costmap_->getSizeInMetersX()) * customized_costmap_.info.width - 0.5);
            y = (int)round(((pose_robot_base_frame.pose.position.y - customized_costmap_.info.origin.position.y)
                            / costmap_->getSizeInMetersY()) * customized_costmap_.info.height - 0.5);

            if ((x >= 0) && (y >= 0) && (x < customized_costmap_.info.width) && (y < customized_costmap_.info.height))
            {
                a_global_plan_map_coordinate.x = x;
                a_global_plan_map_coordinate.y = y;

                global_plan_map_coordinates.push_back(a_global_plan_map_coordinate);
            }
        }

        // add global plan as white pixels
        for(std::vector<geometry_msgs::Point>::iterator it = global_plan_map_coordinates.begin(); it !=
                global_plan_map_coordinates.end(); it++)
        {
            a_global_plan_map_coordinate = *it;
            customized_costmap_.data[a_global_plan_map_coordinate.x + a_global_plan_map_coordinate.y
                                                                      * customized_costmap_.info.width] = 0;
        }

        // add global plan as bright pixels with gradient
        /*int total_plan_pixel_number = global_plan_map_coordinates.size();
        int counter = 0;
        for(std::vector<geometry_msgs::Point>::iterator it = global_plan_map_coordinates.begin(); it !=
         global_plan_map_coordinates.end(); it++) {
            a_global_plan_map_coordinate = *it;
            customized_costmap_.data[a_global_plan_map_coordinate.x + a_global_plan_map_coordinate.y
            * customized_costmap_.info.width] = 50 - round((double)counter/(double)(total_plan_pixel_number-1)*50.0);
            counter++;
        }*/

        // add global blob
        int goal_tolerance_in_pixel = (int)round(goalTolerance / (costmap_->getSizeInMetersX()
                                                                  / costmap_->getSizeInCellsX()));

        geometry_msgs::Point blob_position_map_coordinate;

        bool got_valid_blob_position = false;
        for(std::vector<geometry_msgs::Point>::reverse_iterator it = global_plan_map_coordinates.rbegin(); it !=
                global_plan_map_coordinates.rend(); it++)
        {
            blob_position_map_coordinate = *it;
            if ((blob_position_map_coordinate.x - goal_tolerance_in_pixel >= 0) &&
                (blob_position_map_coordinate.y - goal_tolerance_in_pixel >= 0) &&
                (blob_position_map_coordinate.x + goal_tolerance_in_pixel < customized_costmap_.info.width) &&
                (blob_position_map_coordinate.y + goal_tolerance_in_pixel < customized_costmap_.info.height))
            {
                got_valid_blob_position = true;
                break;
            }
        }

        // goal is is somewhere in the current state representation
        if (got_valid_blob_position)
        {
            int pixel_to_blob_center;
            for (int x = (int)(blob_position_map_coordinate.x - goal_tolerance_in_pixel); x <=
                    blob_position_map_coordinate.x + goal_tolerance_in_pixel; x++)
            {
                for (int y = (int)(blob_position_map_coordinate.y - goal_tolerance_in_pixel); y <=
                        blob_position_map_coordinate.y + goal_tolerance_in_pixel; y++)
                {
                    pixel_to_blob_center = (int)round(sqrt(pow((blob_position_map_coordinate.x - x), 2.0)
                                                           + pow((blob_position_map_coordinate.y  - y), 2.0)));

                    if (pixel_to_blob_center <= goal_tolerance_in_pixel)
                    {
                        customized_costmap_.data[x + y*customized_costmap_.info.width] = 0;
                    }
                }
            }
        }
        else // goal is outside of the current state representation
        {

        }

    }

};
