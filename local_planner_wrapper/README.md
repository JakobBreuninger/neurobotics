Local Planner Wrapper
==============

Local Planner Interface
--------------
To adhere to the local planner interface of `move_base` we *have* to inherit from `nav_core::base_local_planner` and we *have* to implement the following functions:

    LocalPlannerWrapper();

    ~LocalPlannerWrapper();

    void initialize(std::string name, tf::TransformListener* tf, costmap_2d::Costmap2DROS* costmap_ros);

    bool setPlan(const std::vector<geometry_msgs::PoseStamped>& orig_global_plan);

    bool computeVelocityCommands(geometry_msgs::Twist& cmd_vel);

    bool isGoalReached();

So far nothing reasonable has been implemented. The planner just makes the robot drive in circles...

Registering the Plugin
--------------
To create and register the plugin we have to do multiple things:

First, we register the plugin as a `base_local_planner`-plugin, with the following line in the source file:

    PLUGINLIB_EXPORT_CLASS(local_planner_wrapper::LocalPlannerWrapper, nav_core::BaseLocalPlanner)

Then we have to create a `planner_plugin.xml` file in which we describe and name the plugin:

    <library path="lib/liblocal_planner_wrapper">
        <class name="local_planner_wrapper/LocalPlannerWrapper" type="local_planner_wrapper::LocalPlannerWrapper" base_class_type="nav_core::BaseLocalPlanner">
            <description>
                A implementation of an empty local planner
            </description>
        </class>
    </library>

Then we have to include the `planner_plugin.xml` file to the `package.xml` file with:

    <export>
        <nav_core plugin="${prefix}/planner_plugin.xml" />
    </export>

Lastly, we have to create a library of the planner with `CMakeLists.txt` by adding the following line:

    add_library(local_planner_wrapper src/local_planner_wrapper.cpp)

Using the Local Planner
--------------
A description of how to tell `move_base` to use the plugin is included in the `README.md` of the `neuro_stage_sim` package.
