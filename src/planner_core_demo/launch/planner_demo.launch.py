"""Launch one planning demo together with RViz2.

Usage examples:
    ros2 launch planner_core_demo planner_demo.launch.py demo:=kino_astar_demo
    ros2 launch planner_core_demo planner_demo.launch.py demo:=bspline_backend_demo
    ros2 launch planner_core_demo planner_demo.launch.py demo:=ego_planner_demo
    ros2 launch planner_core_demo planner_demo.launch.py demo:=closed_loop_demo
    ros2 launch planner_core_demo planner_demo.launch.py demo:=mapping_demo
    ros2 launch planner_core_demo planner_demo.launch.py demo:=ego_planner_demo rviz:=false
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_share = get_package_share_directory('planner_core_demo')
    default_rviz = os.path.join(pkg_share, 'rviz', 'planner_demo.rviz')

    demo_arg = DeclareLaunchArgument(
        'demo',
        default_value='ego_planner_demo',
        description='Which demo executable to launch (kino_astar_demo | '
                    'bspline_backend_demo | ego_planner_demo | closed_loop_demo | mapping_demo)',
    )
    rviz_arg = DeclareLaunchArgument(
        'rviz', default_value='true', description='Whether to start RViz2')

    demo = LaunchConfiguration('demo')
    use_rviz = LaunchConfiguration('rviz')

    # Publish a static identity transform so the "map" frame exists in the TF tree.
    # All planner demo markers use frame_id="map" and RViz2 Fixed Frame is "map".
    # Some RViz2 / tf2 builds silently drop markers when the frame is missing from TF.
    tf_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='map_tf_broadcaster',
        arguments=['--frame-id', 'world', '--child-frame-id', 'map'],
        output='screen',
    )

    demo_node = Node(
        package='planner_core_demo',
        executable=demo,
        name=demo,
        output='screen',
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', default_rviz],
        condition=IfCondition(use_rviz),
        output='screen',
    )

    return LaunchDescription([demo_arg, rviz_arg, tf_node, demo_node, rviz_node])
