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
from launch.conditions import IfCondition, UnlessCondition
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

    # Robot State Publisher — publishes /robot_description + the robot's TF.
    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        parameters=[{
            "robot_description": Command(["xacro ", urdf_file]),
        }],
        output="screen",
    )

    animate = LaunchConfiguration("animate")
    declare_animate = DeclareLaunchArgument(
        "animate", default_value="true",
        description="Animate the car driving along the planned path. "
                    "false = park it statically at the start pose.")

    # --- animate:=true -> path_player drives the car along /planner/path,
    # publishing BOTH the moving map->base_link TF and /joint_states (steering +
    # wheel spin), so the car visibly drives the path with turning wheels. ---
    path_player = Node(
        package="planner_gazebo_demo",
        executable="path_player",
        name="path_player",
        parameters=[{"play_speed": 0.6, "start_x": -7.0, "start_y": -6.0,
                     "start_yaw": 0.706}],
        condition=IfCondition(animate),
        output="screen",
    )

    # --- animate:=false -> static display: joint_state_publisher fills in the
    # wheel transforms and a static map->base_link parks the car at the start. ---
    joint_state_publisher = Node(
        package="joint_state_publisher",
        executable="joint_state_publisher",
        name="joint_state_publisher",
        condition=UnlessCondition(animate),
        output="screen",
    )
    map_to_base = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="map_to_base_link",
        arguments=["--x", "-7.0", "--y", "-6.0", "--z", "0.0",
                   "--yaw", "0.706", "--pitch", "0.0", "--roll", "0.0",
                   "--frame-id", "map", "--child-frame-id", "base_link"],
        condition=UnlessCondition(animate),
        output="screen",
    )

    # Select the planner executable based on the argument.  The launch argument
    # keeps short names for users, while the installed executables are suffixed
    # with "_planner" in CMakeLists.txt.
    planner_node = Node(
        package="planner_gazebo_demo",
        executable=[planner, "_planner"],
        name=planner,
        parameters=planner_params,
        output="screen",
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=["-d", rviz_config],
        additional_env={
            "LIBGL_ALWAYS_SOFTWARE": "1",
            "QT_X11_NO_MITSHM": "1",
        },
        output="screen",
    )

    return LaunchDescription([
        declare_map,
        declare_planner,
        declare_rviz,
        declare_animate,
        robot_state_publisher,
        path_player,
        joint_state_publisher,
        map_to_base,
        planner_node,
        rviz_node,
    ])
