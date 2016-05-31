#ifndef NEURO_FAKE_RECOVERY_H_
#define NEURO_FAKE_RECOVERY_H_
#include <nav_core/recovery_behavior.h>
#include <costmap_2d/costmap_2d_ros.h>
#include <tf/transform_listener.h>
#include <ros/ros.h>
#include <base_local_planner/costmap_model.h>
#include <std_msgs/Bool.h>

namespace neuro_fake_recovery{
    /**
     * @class KeepGoingRecovery
     * @brief A recovery behavior that beams the robot to a new location with a new goal point if its stuck
     */
    class NeuroFakeRecovery : public nav_core::RecoveryBehavior
    {
        public:
            /**
             * @brief  Constructor
             * @param
             * @return
             */
            NeuroFakeRecovery();

            /**
             * @brief  Initialization function for the KeepGoingRecovery recovery behavior
             * @param tf A pointer to a transform listener
             * @param global_costmap A pointer to the global_costmap used by the navigation stack
             * @param local_costmap A pointer to the local_costmap used by the navigation stack
             */
            void initialize(std::string name, tf::TransformListener* tf,
                            costmap_2d::Costmap2DROS* global_costmap, costmap_2d::Costmap2DROS* local_costmap);

            /**
             * @brief  Run the KeepGoingRecovery recovery behavior.
             */
            void runBehavior();

            /**
             * @brief  Destructor
             */
            ~NeuroFakeRecovery();

        private:
            std::string name_;

            bool initialized_;

            // Publisher to the stage_sim_bot
            ros::Publisher state_pub_;
    };
};
#endif