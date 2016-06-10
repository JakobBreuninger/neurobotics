#include "ros/ros.h"
#include "std_msgs/Bool.h"
#include "geometry_msgs/Pose.h"
#include "geometry_msgs/PoseWithCovarianceStamped.h"
#include "geometry_msgs/PoseStamped.h"

// Publisher and subscribers
ros::Publisher stage_pub;
ros::Publisher move_base_goal_pub;

// Uncomment when using real amcl localization
// ros::Publisher move_base_pose_pub;

void botCallback(const std_msgs::Bool new_round)
{
    if(new_round.data)
    {
        // Set some random points and push them into a vector of points
        /*geometry_msgs::Pose x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16, x_17,
                            x_18, x_19, x_20, x_21, x_22, x_23, x_24, x_25, x_26, x_27, x_28, x_29, x_30;
        std::vector<geometry_msgs::Pose> poses;
        x_1.position.x = 2.06546545029;
        x_1.position.y = 4.28590583801;
        poses.push_back(x_1);

        x_2.position.x = 5.44342184067;
        x_2.position.y = 0.880249977112;
        poses.push_back(x_2);

        x_3.position.x = 1.84457874298;
        x_3.position.y = 6.83638143539;
        poses.push_back(x_3);

        x_4.position.x = 6.35693025589;
        x_4.position.y = 3.92254328728;
        poses.push_back(x_4);

        x_5.position.x = 7.59819555283;
        x_5.position.y = 3.7883439064;
        poses.push_back(x_5);

        x_6.position.x = 8.85293960571;
        x_6.position.y = 1.24048757553;
        poses.push_back(x_6);

        x_7.position.x = 7.17635631561;
        x_7.position.y = 5.95679283142;
        poses.push_back(x_7);

        x_8.position.x = 6.53284931183;
        x_8.position.y = 9.24268722534;
        poses.push_back(x_8);

        x_9.position.x = 2.5139811039;
        x_9.position.y = 4.14342975616;
        poses.push_back(x_9);

        x_10.position.x = 7.66414833069;
        x_10.position.y = 7.6863527298;
        poses.push_back(x_10);

        x_11.position.x = 7.77705907822;
        x_11.position.y = 7.32463550568;
        poses.push_back(x_11);

        x_12.position.x = -1.1419 + 2.0;
        x_12.position.y = 2.525 + 2.0;
        poses.push_back(x_12);

        x_13.position.x = 0.4474 + 2.0;
        x_13.position.y = 2.723 + 2.0;
        poses.push_back(x_13);

        x_14.position.x = 0.767 + 2.0;
        x_14.position.y = 5.6718 + 2.0;
        poses.push_back(x_14);

        x_15.position.x = 3.092 + 2.0;
        x_15.position.y = 4.637 + 2.0;
        poses.push_back(x_15);

        x_16.position.x = -0.941 + 2.0;
        x_16.position.y = 6.898 + 2.0;
        poses.push_back(x_16);

        x_17.position.x = 0.673 + 2.0;
        x_17.position.y = 5.64 + 2.0;
        poses.push_back(x_17);

        x_18.position.x = 7.022 + 2.0;
        x_18.position.y = -1.060 + 2.0;
        poses.push_back(x_18);

        x_19.position.x = 4.614 + 2.0;
        x_19.position.y = -0.115 + 2.0;
        poses.push_back(x_19);

        x_20.position.x = 3.03 + 2.0;
        x_20.position.y = 2.57 + 2.0;
        poses.push_back(x_20);

        x_21.position.x = 5.57 + 2.0;
        x_21.position.y = 1.33 + 2.0;
        poses.push_back(x_21);

        x_22.position.x = 4.80 + 2.0;
        x_22.position.y = -0.98 + 2.0;
        poses.push_back(x_22);

        x_23.position.x = 3.62 + 2.0;
        x_23.position.y = 7.04 + 2.0;
        poses.push_back(x_23);

        x_24.position.x = 4.242 + 2.0;
        x_24.position.y = 1.864 + 2.0;
        poses.push_back(x_24);

        x_25.position.x = -0.06 + 2.0;
        x_25.position.y = 4.875 + 2.0;
        poses.push_back(x_25);

        x_26.position.x = 3.95 + 2.0;
        x_26.position.y = 5.33 + 2.0;
        poses.push_back(x_26);

        x_27.position.x = 7.21 + 2.0;
        x_27.position.y = 4.73 + 2.0;
        poses.push_back(x_27);

        x_28.position.x = 0.28 + 2.0;
        x_28.position.y = 0.96 + 2.0;
        poses.push_back(x_28);

        x_29.position.x = 4.58 + 2.0;
        x_29.position.y = 1.23 + 2.0;
        poses.push_back(x_29);

        x_30.position.x = 7.19 + 2.0;
        x_30.position.y = 7.05 + 2.0;
        poses.push_back(x_30);

        // Now randomly choose two points to use as start pose and goal pose and make sure they are different ones
        srand((unsigned int)time(NULL));
        unsigned long start = rand() % poses.size();
        unsigned long goal = rand() % poses.size();
        while (start == goal)
        {
            goal = rand() % poses.size();
        }*/

        // ROS_ERROR("Start: %d, Goal: %d", (int)start, (int)goal);

        // Get x and y coordinates and orientation for start point
        double x = (double)(rand() % 130)/100.0 - 0.5 + 2.0;
        double y = (double)(rand() % 170)/100.0 + 2.0;
        double o = (double)(rand() % 400)/100.0;

        // Send new position to stage
        geometry_msgs::Pose pose;
        pose.position.z = 0.0;
        pose.position.x = x;
        pose.position.y = y;
        pose.orientation.z = 1.0;
        pose.orientation.w = o;
        stage_pub.publish(pose);

        // Uncomment when actually using amcl localization
        /*// Send new position to move_base
        geometry_msgs::PoseWithCovarianceStamped pose_with_co;
        pose_with_co.pose.pose.orientation.z = 1.0;
        pose_with_co.pose.pose.position = poses.at(start).position;
        pose_with_co.header.frame_id = "map";
        move_base_pose_pub.publish(pose_with_co);*/

        // Make sure that the global planner is aware of the new position
        ros::Rate r(1);
        r.sleep();

        // Get x and y coordinates and orientation for start point + 2.0 for coordinate transform...
        // TODO: automate the transform
        x = (double)(rand() % 130)/100.0 - 0.5 + 2.0;
        y = (double)(rand() % 170)/100.0 + 2.0;
        o = (double)(rand() % 400)/100.0;

        // Send new goal position to move_base
        geometry_msgs::PoseStamped pose_stamped;
        pose_stamped.pose.position.z = 0.0;
        pose_stamped.pose.position.x = x;
        pose_stamped.pose.position.y = y;
        pose_stamped.pose.orientation.z = 1.0;
        pose_stamped.pose.orientation.w = o;
        pose_stamped.header.frame_id = "map";
        move_base_goal_pub.publish(pose_stamped);
    }
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "neuro_training_bot");

    ros::NodeHandle n;

    // Subscribers
    ros::Subscriber sub_planner = n.subscribe("/move_base/NeuroLocalPlannerWrapper/new_round", 1000, botCallback);
    ros::Subscriber sub_recovery = n.subscribe("/move_base/neuro_fake_recovery/new_round", 1000, botCallback);

    // Publishers
    stage_pub = n.advertise<geometry_msgs::Pose>("neuro_stage_ros/set_pose", 1);
    move_base_goal_pub = n.advertise<geometry_msgs::PoseStamped>("move_base_simple/goal", 1);

    // Uncomment when using real amcl localization
    //move_base_pose_pub = n.advertise<geometry_msgs::PoseWithCovarianceStamped>("initialpose", 1);

    // Make sure that the global planner is aware of the new position
    ros::Rate r(0.5);
    r.sleep();

    // Get x and y coordinates and orientation for start point
    double x = (double)(rand() % 130)/100.0 - 0.5 + 2.0;
    double y = (double)(rand() % 170)/100.0 + 2.0;
    double o = (double)(rand() % 400)/100.0;

    // Send new position to stage
    geometry_msgs::PoseStamped pose;
    pose.pose.position.z = 0.0;
    pose.pose.position.x = x;
    pose.pose.position.y = y;
    pose.pose.orientation.z = 1.0;
    pose.pose.orientation.w = o;
    pose.header.frame_id = "map";
    move_base_goal_pub.publish(pose);

    ros::spin();

    return 0;
}