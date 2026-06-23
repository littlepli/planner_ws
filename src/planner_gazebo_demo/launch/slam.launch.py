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
    ExecuteProcess,
    IncludeLaunchDescription,
    RegisterEventHandler,
    TimerAction,
)
from launch.event_handlers import OnProcessStart
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory("planner_gazebo_demo")

    world_file = os.path.join(pkg_share, "worlds", "planner_world.sdf")
    urdf_file = os.path.join(pkg_share, "urdf", "ackermann_car.xacro")
    rviz_config = os.path.join(pkg_share, "rviz", "slam.rviz")
    controller_config = os.path.join(pkg_share, "config", "ackermann_control.yaml")

    # --- Gazebo ---
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
    spawn_robot = Node(
        package="gazebo_ros",
        executable="spawn_entity.py",
        arguments=[
            "-entity", "ackermann_car",
            "-topic", "robot_description",
            "-x", "-7.0", "-y", "-6.0", "-z", "0.3",
            "-Y", "0.706",
        ],
        output="screen",
    )

    # --- Load controllers ---
    load_joint_broadcaster = ExecuteProcess(
        cmd=["ros2", "control", "load_controller", "--set-active",
             "joint_state_broadcaster"],
        output="screen",
    )

    load_ackermann_controller = ExecuteProcess(
        cmd=["ros2", "control", "load_controller", "--set-active",
             "ackermann_steering_controller"],
        output="screen",
    )

    # --- SLAM Toolbox (Gmapping mode) ---
    slam_params = {
        "use_sim_time": True,
        "solver_mode": "2D",
        "slam_toolbox_node": {
            "ros__parameters": {
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
                "publish_map_scan": True,
                "loop_search_maximum_distance": 10.0,
                "loop_match_minimum_chain_size": 10,
                "loop_match_maximum_variance": 0.4,
                "loop_match_minimum_response_coarse": 0.7,
                "loop_match_minimum_response_fine": 0.8,
            }
        },
    }

    slam = Node(
        package="slam_toolbox",
        executable="sync_slam_toolbox_node",
        name="slam_toolbox",
        parameters=[slam_params],
        output="screen",
        remappings=[("/scan", "/scan")],
    )

    # --- RViz ---
    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=["-d", rviz_config],
        parameters=[{"use_sim_time": True}],
        output="screen",
    )

    return LaunchDescription([
        gazebo,
        robot_state_publisher,
        spawn_robot,
        TimerAction(period=5.0, actions=[load_joint_broadcaster]),
        TimerAction(period=7.0, actions=[load_ackermann_controller]),
        TimerAction(period=8.0, actions=[slam]),
        TimerAction(period=9.0, actions=[rviz]),
    ])