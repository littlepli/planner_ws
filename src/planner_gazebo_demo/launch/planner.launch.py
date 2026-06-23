"""
Planner launch file: loads a saved PGM+YAML map and runs one of three
Ackermann path planners (kino_astar, bspline, ego), visualised in RViz
with the OccupancyGrid, Path, and TF displays.

Usage:
  ros2 launch planner_gazebo_demo planner.launch.py planner:=kino_astar
  ros2 launch planner_gazebo_demo planner.launch.py planner:=bspline
  ros2 launch planner_gazebo_demo planner.launch.py planner:=ego
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory("planner_gazebo_demo")

    default_map = os.path.join(pkg_share, "maps", "planner_map.yaml")
    default_rviz = os.path.join(pkg_share, "rviz", "planner.rviz")
    urdf_file = os.path.join(pkg_share, "urdf", "ackermann_car.xacro")

    map_yaml = LaunchConfiguration("map_yaml")
    planner = LaunchConfiguration("planner")
    rviz_config = LaunchConfiguration("rviz_config")

    declare_map = DeclareLaunchArgument(
        "map_yaml", default_value=default_map,
        description="Path to the map YAML file")
    declare_planner = DeclareLaunchArgument(
        "planner", default_value="kino_astar",
        description="Which planner to run: kino_astar, bspline, or ego",
        choices=["kino_astar", "bspline", "ego"])
    declare_rviz = DeclareLaunchArgument(
        "rviz_config", default_value=default_rviz,
        description="Path to RViz config file")

    # Common planner parameters
    planner_params = [{
        "map_yaml": map_yaml,
        "start_x": -7.0,
        "start_y": -6.0,
        "start_theta": 0.706,  # ~atan2(6,7)
        "goal_x": 7.0,
        "goal_y": 6.0,
    }]

    # Robot State Publisher — publishes /robot_description for RViz RobotModel
    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        parameters=[{
            "robot_description": Command(["xacro ", urdf_file]),
        }],
        output="screen",
    )

    # Select the planner executable based on the argument
    planner_node = Node(
        package="planner_gazebo_demo",
        executable=planner,
        name=planner,
        parameters=planner_params,
        output="screen",
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=["-d", rviz_config],
        output="screen",
    )

    return LaunchDescription([
        declare_map,
        declare_planner,
        declare_rviz,
        robot_state_publisher,
        planner_node,
        rviz_node,
    ])
