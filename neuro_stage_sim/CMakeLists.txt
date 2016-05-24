cmake_minimum_required(VERSION 2.8.3)
project(neuro_stage_sim)

find_package(catkin REQUIRED COMPONENTS roscpp rospy std_msgs genmsg nav_core pluginlib)

include_directories(include ${catkin_INCLUDE_DIRS})

catkin_package(
        INCLUDE_DIRS include
        LIBRARIES keep_going_recovery
        CATKIN_DEPENDS
        roscpp
        pluginlib
)

add_executable(neuro_training_bot src/neuro_training_bot.cpp)
target_link_libraries(neuro_training_bot ${catkin_LIBRARIES})

add_library(neuro_fake_recovery src/neuro_fake_recovery.cpp)
target_link_libraries(neuro_fake_recovery ${catkin_LIBRARIES})