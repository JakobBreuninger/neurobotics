cmake_minimum_required(VERSION 2.8.3)
project(neuro_deep_planner)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  neuro_local_planner_wrapper
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)
