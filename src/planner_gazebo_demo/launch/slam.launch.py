"""
SLAM launch file: Gazebo + Ackermann car + slam_toolbox (Gmapping) + RViz.

Usage:
  ros2 launch planner_gazebo_demo slam.launch.py

After Gazebo and RViz appear, open another terminal and run the teleop node:
  ros2 run planner_gazebo_demo ackermann_teleop

Drive the car around to build the map. When done, save with:
  ros2 run nav2_map_server map_saver_cli -f src/planner_gazebo_demo/maps/planner_map
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    SetEnvironmentVariable,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory("planner_gazebo_demo")

    # --- WSL2 / headless-GL software rendering ---
    # Set at launch scope (SetEnvironmentVariable) so EVERY child process inherits
    # it — crucially Gazebo (gzserver/gzclient), which is started via the included
    # gazebo.launch.py below and therefore cannot be reached by a per-Node
    # `additional_env`. On WSL2 / remote-X without a usable hardware GL driver,
    # Gazebo Classic and RViz2 otherwise crash on start.
    #   LIBGL_ALWAYS_SOFTWARE=1  -> force software (llvmpipe) OpenGL
    #   QT_X11_NO_MITSHM=1       -> disable X11 MIT-SHM (Qt GUIs)   [matches rviz2_sw]
    #   SVGA_VGPU10=0            -> disable VMware vGPU10 path; needed by Gazebo
    #                              Classic / OGRE under WSL software GL.
    env_libgl = SetEnvironmentVariable("LIBGL_ALWAYS_SOFTWARE", "1")
    env_mitshm = SetEnvironmentVariable("QT_X11_NO_MITSHM", "1")
    env_svga = SetEnvironmentVariable("SVGA_VGPU10", "0")

    world_file = os.path.join(pkg_share, "worlds", "planner_world.sdf")
    urdf_file = os.path.join(pkg_share, "urdf", "ackermann_car.xacro")
    rviz_config = os.path.join(pkg_share, "rviz", "slam.rviz")
    controller_config = os.path.join(pkg_share, "config", "ackermann_control.yaml")

    # --- Gazebo ---
    # On WSL2 with software-rendered OpenGL the Gazebo GUI (gzclient) is very
    # heavy and starves gzserver during startup, so /spawn_entity is often not
    # ready within the spawn timeout and the robot fails to spawn (no robot ->
    # no controller_manager -> no SLAM). Default to HEADLESS (gui:=false) here
    # and watch everything in RViz instead. Pass gui:=true to show the Gazebo
    # window (only advisable on a machine with a real GPU).
    gui = LaunchConfiguration("gui")
    declare_gui = DeclareLaunchArgument(
        "gui", default_value="false",
        description="Launch the Gazebo client GUI (gzclient). "
                    "Keep false on WSL2 / software GL for a reliable startup.")

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("gazebo_ros"),
                "launch",
                "gazebo.launch.py",
            )
        ),
        launch_arguments={
            "world": world_file,
            "verbose": "true",
            "gui": gui,
        }.items(),
    )

    # --- Robot State Publisher (process xacro → URDF) ---
    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[{
            "use_sim_time": True,
            "robot_description": Command(["xacro ", urdf_file]),
        }],
        output="screen",
    )

    # --- Spawn robot in Gazebo ---
    # Long --timeout so a slow (software-rendered / WSL) gzserver still has time
    # to bring up the /spawn_entity service before this node gives up.
    spawn_robot = Node(
        package="gazebo_ros",
        executable="spawn_entity.py",
        arguments=[
            "-entity", "ackermann_car",
            "-topic", "robot_description",
            "-x", "-7.0", "-y", "-6.0", "-z", "0.1",
            "-Y", "0.706",
            "-timeout", "120.0",
        ],
        output="screen",
    )

    # --- Load + activate controllers via controller_manager `spawner` ---
    # NOTE: previously this used `ros2 control load_controller`, which requires
    # the `ros2controlcli` package — if that package is not installed the command
    # dies with "invalid choice: 'control'" and NO controller is activated, which
    # cascades into: no /joint_states, no odom->base_link TF, slam_toolbox drops
    # every scan, no map is built, and map_saver fails.  The `spawner` executable
    # lives in `controller_manager` (already a dependency) and waits for the
    # controller_manager to come up, so it works without ros2controlcli.
    load_joint_broadcaster = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster",
                   "--controller-manager", "/controller_manager"],
        output="screen",
    )

    load_ackermann_controller = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["ackermann_steering_controller",
                   "--controller-manager", "/controller_manager"],
        output="screen",
    )

    # --- TF odometry relay ---
    # The ackermann_steering_controller publishes its odom->base_link transform on
    # its PRIVATE topic ~/tf_odometry (NOT on /tf), so the `odom` frame never joins
    # the global TF tree and slam_toolbox drops every scan. This relay republishes
    # ~/tf_odometry onto /tf to complete the map->odom->base_link->lidar chain.
    tf_odometry_relay = Node(
        package="planner_gazebo_demo",
        executable="tf_odometry_relay",
        name="tf_odometry_relay",
        # Restamp to current sim time (future_offset=0 -> accurate odom). slam's
        # transform_timeout absorbs the small relay latency.
        parameters=[{"use_sim_time": True, "restamp": True, "future_offset": 0.0}],
        output="screen",
    )

    # --- SLAM Toolbox (Gmapping mode) ---
    # NOTE: when passing parameters INLINE to a launch Node, they must be a FLAT
    # dict. The earlier nested form {"slam_toolbox_node": {"ros__parameters": {..}}}
    # is the *YAML-file* layout; inline, launch_ros flattens it to bogus parameter
    # names like "slam_toolbox_node.ros__parameters.base_frame", so the real
    # `base_frame` stayed at its DEFAULT ("base_footprint"). This robot has
    # `base_link` and no `base_footprint`, so slam_toolbox could never transform
    # base_frame->odom -> "Failed to compute odom pose" forever and no map built.
    # Flattening the dict makes base_frame=base_link (and the rest) actually apply.
    slam_params = {
        "use_sim_time": True,
        "base_frame": "base_link",
        "odom_frame": "odom",
        "map_frame": "map",
        "mode": "mapping",
        "scan_topic": "/scan",
        "resolution": 0.05,
        "max_laser_range": 30.0,
        "minimum_time_interval": 0.5,
        "transform_publish_period": 0.05,
        "map_update_interval": 1.0,
        # "transform_timeout": 0.8,
        "transform_timeout": 1.0,
        "minimum_travel_distance": 0.3,
        "loop_search_maximum_distance": 10.0,
        "loop_match_minimum_chain_size": 10,
        "loop_match_maximum_variance": 0.4,
        "loop_match_minimum_response_coarse": 0.7,
        "loop_match_minimum_response_fine": 0.8,
        "min_laser_range": 0.2,
    }

    slam = Node(
        package="slam_toolbox",
        executable="sync_slam_toolbox_node",
        name="slam_toolbox",
        parameters=[slam_params],
        output="screen",
        remappings=[("/scan", "/scan")],
    )

    # --- Auto-explore (optional hands-free mapping) ---
    # explore:=true starts a reactive wander/obstacle-avoidance driver so SLAM
    # fills in the map without any manual driving.
    explore = LaunchConfiguration("explore")
    declare_explore = DeclareLaunchArgument(
        "explore", default_value="false",
        description="Auto-drive the car (reactive exploration) to build the map "
                    "hands-free. Leave false to drive manually with ackermann_teleop.")
    auto_explore = Node(
        package="planner_gazebo_demo",
        executable="auto_explore",
        name="auto_explore",
        condition=IfCondition(explore),
        output="screen",
    )

    # --- RViz ---
    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=["-d", rviz_config],
        parameters=[{"use_sim_time": True}],
        additional_env={
            "LIBGL_ALWAYS_SOFTWARE": "1",
            "QT_X11_NO_MITSHM": "1",
        },
        output="screen",
    )

    return LaunchDescription([
        # Env vars first, so Gazebo (and everything after) inherits software GL.
        env_libgl,
        env_mitshm,
        env_svga,
        declare_gui,
        declare_explore,
        gazebo,
        robot_state_publisher,
        spawn_robot,
        TimerAction(period=5.0, actions=[load_joint_broadcaster]),
        TimerAction(period=7.0, actions=[load_ackermann_controller]),
        # TimerAction(period=8.0, actions=[slam, tf_odometry_relay]),
        TimerAction(period=8.0, actions=[tf_odometry_relay]),
        TimerAction(period=10.0, actions=[slam]),
        TimerAction(period=12.0, actions=[rviz]),
        # Start exploring after controllers + SLAM are up.
        TimerAction(period=14.0, actions=[auto_explore]),
        # TimerAction(period=9.0, actions=[rviz]),
    ])
