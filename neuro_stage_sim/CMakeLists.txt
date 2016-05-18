cmake_minimum_required(VERSION 2.8.3)
project(neuro_stage_sim)

find_package(catkin REQUIRED COMPONENTS roscpp rospy std_msgs genmsg)

include_directories(include ${catkin_INCLUDE_DIRS})

catkin_package()

add_executable(neuro_stage_sim_bot src/neuro_stage_sim_bot.cpp)
target_link_libraries(neuro_stage_sim_bot ${catkin_LIBRARIES})