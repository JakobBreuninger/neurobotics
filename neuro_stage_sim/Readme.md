Instructions to Run Simulation
==============

TurtleBot Dependencies
--------------

First you need to install the following debs for *TurtleBot*:

    sudo apt-get install ros-indigo-turtlebot ros-indigo-turtlebot-apps ros-indigo-turtlebot-interactions ros-indigo-turtlebot-simulator ros-indigo-kobuki-ftdi ros-indigo-rocon-remocon ros-indigo-rocon-qt-library ros-indigo-ar-track-alvar-msgs

Install the Package
--------------

Download the zipped repository (you could also clone it) and place the `neuro_stage_sim` folder into your `/catkin_ws/src/` directory:

    https://github.com/JakobBreuninger/neurobotics/archive/master.zip

Go into your `/catkin_ws/` directory and compile the new package with:

    catkin_make


While still in your `/catkin_ws/` directory, make sure ROS knows where to find the new package with:

    source devel/setup.bash

Run the Simulation
--------------

Simply run the simulation with the `neuro_stage_sim.launch` file:

    roslaunch neuro_stage_sim neuro_stage_sim.launch

Click the `2D Nav Goal` button in *RViz* to send the robot around the environment.

By default the robot is put into the `maze` environment. If you want to switch to the `robopark_plan` environment, simply go into the `neuro_stage_sim.launch` file and change lines `16` and `17` to:

    <arg name="map_file"       default="$(find neuro_stage_sim)/maps/robopark_plan.yaml"/>
    <arg name="world_file"     default="$(find neuro_stage_sim)/maps/stage/robopark_plan.world"/>

By default *stage* is set to run 3 times as fast as real-time. To change this go into `maze.world` or `robopark_plan.world` and change the parameter `speedup`

Run the Local Planner Plugin
--------------

In the `move_base.launch.xml` file we can use the dummy local planner plugin by uncommenting the following line:

    <!-- param name="base_local_planner" value="local_planner_wrapper/LocalPlannerWrapper"/ -->

And by commenting the following line right above:

    <param name="DWAPlannerROS/global_frame_id" value="$(arg odom_frame_id)"/>
