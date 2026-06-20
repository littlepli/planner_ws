"""Launch Gazebo + Ackermann car + planner node + RViz.

Usage:
    ros2 launch planner_gazebo_demo gazebo_demo.launch.py scenario:=dense

"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, RegisterEventHandler
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, Command
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_share = get_package_share_directory('planner_gazebo_demo')
    planner_share = get_package_share_directory('planner_core_demo')

    # --------- arguments ----------
    scenario_arg = DeclareLaunchArgument('scenario', default_value='dense',
        description='Obstacle layout: dense | clustered | narrow')
    use_rviz_arg = DeclareLaunchArgument('rviz', default_value='true',
        description='Open RViz2')

    scenario = LaunchConfiguration('scenario')
    use_rviz = LaunchConfiguration('rviz')

    # --------- robot description (URDF via xacro) ----------
    robot_description = {'robot_description': Command(
        ['xacro ', os.path.join(pkg_share, 'urdf', 'ackermann_car.xacro')])}

    # --------- Gazebo server + client ----------
    gazebo_server = ExecuteProcess(
        cmd=['gzserver', '-s', 'libgazebo_ros2_control.so',
             os.path.join(pkg_share, 'worlds', 'planner_world.sdf')],
        output='screen')

    gazebo_client = ExecuteProcess(
        cmd=['gzclient'],
        output='screen')

    # --------- robot_state_publisher ----------
    robot_state_pub = Node(
        package='robot_state_publisher', executable='robot_state_publisher',
        parameters=[robot_description])

    # --------- controller manager ----------
    controller_manager = Node(
        package='controller_manager', executable='ros2_control_node',
        parameters=[robot_description,
                    os.path.join(pkg_share, 'config', 'ackermann_control.yaml')],
        remappings=[('~/robot_description', '/robot_description')])

    # Activate controllers after the manager is up
    jsb_spawner = Node(
        package='controller_manager', executable='spawner',
        arguments=['joint_state_broadcaster', '--controller-manager', '/controller_manager'])

    ackermann_spawner = Node(
        package='controller_manager', executable='spawner',
        arguments=['ackermann_steering_controller', '--controller-manager', '/controller_manager'])

    # --------- planner node ----------
    planner_node = Node(
        package='planner_gazebo_demo', executable='gazebo_planner_node',
        name='gazebo_planner_node', output='screen',
        parameters=[{'scenario': scenario}])

    # --------- RViz2 (reuse planner_core_demo config) ----------
    rviz_config = os.path.join(planner_share, 'rviz', 'planner_demo.rviz')
    rviz_node = Node(
        package='rviz2', executable='rviz2', name='rviz2',
        arguments=['-d', rviz_config],
        condition=IfCondition(use_rviz), output='screen')

    return LaunchDescription([
        scenario_arg, use_rviz_arg,
        gazebo_server, gazebo_client,
        robot_state_pub, controller_manager,
        jsb_spawner, ackermann_spawner,
        planner_node, rviz_node,
    ])