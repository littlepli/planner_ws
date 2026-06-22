import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg = 'planner_gazebo_demo'

    # ── Paths ─────────────────────────────────────────────────────
    pkg_share = get_package_share_directory(pkg)
    urdf_file = os.path.join(pkg_share, 'urdf', 'ackermann_car.xacro')
    world_file = os.path.join(pkg_share, 'worlds', 'planner_world.sdf')
    controller_config = os.path.join(pkg_share, 'config', 'ackermann_control.yaml')

    # ── Launch args ────────────────────────────────────────────────
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    use_rviz = LaunchConfiguration('use_rviz', default='false')
    headless = LaunchConfiguration('headless_gazebo', default='false')

    # ── Robot description (xacro → urdf) ──────────────────────────
    robot_description = {
        'robot_description': Command(['xacro ', urdf_file, ' use_sim:=true'])
    }

    # ── 1. Gazebo (with or without GUI) ───────────────────────────
    gazebo_gui = ExecuteProcess(
        cmd=['gazebo', '--verbose',
             '-s', 'libgazebo_ros_init.so',
             '-s', 'libgazebo_ros_factory.so', world_file],
        output='screen',
        condition=UnlessCondition(headless))

    gazebo_headless = ExecuteProcess(
        cmd=['gazebo', '--verbose', '-u',
             '-s', 'libgazebo_ros_init.so',
             '-s', 'libgazebo_ros_factory.so', world_file],
        output='screen',
        condition=IfCondition(headless))

    # ── 2. Spawn the Ackermann car (delayed for robot_description) ──
    spawn_entity = TimerAction(
        period=2.0,
        actions=[Node(
            package='gazebo_ros', executable='spawn_entity.py',
            arguments=['-entity', 'ackermann_car', '-topic', 'robot_description',
                       '-x', '0', '-y', '0', '-z', '0.3'],
            output='screen')])

    # ── 3. robot_state_publisher ──────────────────────────────────
    robot_state_pub = Node(
        package='robot_state_publisher', executable='robot_state_publisher',
        parameters=[robot_description,
                    {'publish_robot_description': True},
                    {'use_sim_time': use_sim_time}],
        output='screen')

    # ── 4. Controller spawners (Gazebo internal controller_manager) ──
    joint_state_broadcaster_spawner = TimerAction(
        period=3.0,
        actions=[Node(
            package='controller_manager', executable='spawner',
            arguments=['joint_state_broadcaster', '--param-file', controller_config],
            output='screen'
        )]
    )

    ackermann_controller_spawner = TimerAction(
        period=5.0,
        actions=[Node(
            package='controller_manager', executable='spawner',
            arguments=['ackermann_steering_controller', '--param-file', controller_config],
            output='screen'
        )]
    )

    # ── 5. RViz2 (optional, default off) ──────────────────────────
    rviz = Node(
        package='rviz2', executable='rviz2',
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen',
        condition=IfCondition(use_rviz))

    # ── 6. Ackermann teleop node ──────────────────────────────────
    teleop = Node(
        package=pkg, executable='ackermann_teleop',
        name='ackermann_teleop', output='screen',
        prefix='xterm -e')

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true',
                              description='Use simulation (Gazebo) time'),
        DeclareLaunchArgument('use_rviz', default_value='false',
                              description='Start RViz2 for visualization'),
        DeclareLaunchArgument('headless_gazebo', default_value='false',
                              description='Run Gazebo without GUI (headless)'),

        gazebo_gui,
        gazebo_headless,
        robot_state_pub,
        spawn_entity,
        joint_state_broadcaster_spawner,
        ackermann_controller_spawner,
        rviz,
        teleop,
    ])