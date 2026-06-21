"""Launch Gazebo + Ackermann car + planner node + RViz.

Usage:
    ros2 launch planner_gazebo_demo gazebo_demo.launch.py scenario:=dense

Key points:
  - The ``libgazebo_ros2_control.so`` plugin is loaded automatically when the
    URDF is spawned (it is declared inside the URDF ``<gazebo>`` block), so we
    do **not** pass ``-s libgazebo_ros2_control.so`` to ``gzserver`` and we do
    **not** start a separate ``ros2_control_node``.
  - Controller spawners are delayed until ``spawn_entity`` finishes so that the
    controller manager inside Gazebo is ready.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, RegisterEventHandler, TimerAction
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration, Command
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('planner_gazebo_demo')

    # --------- arguments ----------
    scenario_arg = DeclareLaunchArgument('scenario', default_value='dense',
        description='Obstacle layout: dense | clustered | narrow')
    use_rviz_arg = DeclareLaunchArgument('rviz', default_value='true',
        description='Open RViz2')
    gazebo_gui_arg = DeclareLaunchArgument('gazebo_gui', default_value='true',
        description='Show Gazebo GUI (set to false for headless)')

    scenario = LaunchConfiguration('scenario')
    use_rviz = LaunchConfiguration('rviz')
    gazebo_gui = LaunchConfiguration('gazebo_gui')

    # --------- robot description (URDF via xacro) ----------
    robot_description = {'robot_description': Command(
        ['xacro ', os.path.join(pkg_share, 'urdf', 'ackermann_car.xacro')])}

    # --------- Gazebo server ----------
    # NOTE: Do NOT pass -s libgazebo_ros2_control.so here. The plugin is loaded
    # automatically when the URDF (which contains the <gazebo> plugin block) is
    # spawned into the world.
    gazebo_server = ExecuteProcess(
        cmd=['gzserver', '--verbose',
             os.path.join(pkg_share, 'worlds', 'planner_world.sdf')],
        output='screen')

    # Gazebo client (GUI), optional
    gazebo_client = ExecuteProcess(
        cmd=['gzclient'],
        condition=IfCondition(gazebo_gui),
        output='screen')

    # --------- robot_state_publisher ----------
    robot_state_pub = Node(
        package='robot_state_publisher', executable='robot_state_publisher',
        parameters=[robot_description, {'use_sim_time': True}])

    # Spawn the robot model into the Gazebo world.  The libgazebo_ros2_control
    # plugin inside the URDF will create the controller_manager.
    # NOTE: We delay spawning by 8 seconds to give gzserver time to fully start,
    # especially on WSL2 where gzserver startup is slower.  If spawn_entity runs
    # before gzserver is ready, the model fails to spawn and the controller
    # plugin never loads, resulting in missing odom->base_link TF.
    spawn_entity = Node(
        package='gazebo_ros', executable='spawn_entity.py',
        arguments=['-topic', 'robot_description', '-entity', 'ackermann_car',
                   '-x', '-7.0', '-y', '-6.0', '-z', '1.0'],
        output='screen')

    # Delay spawn_entity until gzserver is fully ready
    delayed_spawn = TimerAction(period=8.0, actions=[spawn_entity])

    # --------- controller spawners ----------
    # These talk to the controller_manager that lives inside Gazebo (created by
    # the libgazebo_ros2_control plugin).  We delay them until spawn_entity exits.
    jsb_spawner = Node(
        package='controller_manager', executable='spawner',
        arguments=['joint_state_broadcaster', '--controller-manager', '/controller_manager'],
        output='screen')

    ackermann_spawner = Node(
        package='controller_manager', executable='spawner',
        arguments=['ackermann_steering_controller', '--controller-manager', '/controller_manager'],
        output='screen')

    # Delay spawners until the model has been spawned into Gazebo
    delay_jsb = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=spawn_entity,
            on_exit=[jsb_spawner]))

    delay_ackermann = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=jsb_spawner,
            on_exit=[ackermann_spawner]))

    # --------- planner node ----------
    planner_node = Node(
        package='planner_gazebo_demo', executable='gazebo_planner_node',
        name='gazebo_planner_node', output='screen',
        parameters=[{'scenario': scenario, 'use_sim_time': True}])

    # Delay planner until ackermann controller is active
    delay_planner = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=ackermann_spawner,
            on_exit=[planner_node]))

    # --------- RViz2 ----------
    rviz_config = os.path.join(pkg_share, 'rviz', 'gazebo.rviz')
    rviz_node = Node(
        package='rviz2', executable='rviz2', name='rviz2',
        arguments=['-d', rviz_config],
        condition=IfCondition(use_rviz), output='screen')

    return LaunchDescription([
        scenario_arg, use_rviz_arg, gazebo_gui_arg,
        gazebo_server, gazebo_client,
        robot_state_pub, delayed_spawn,
        delay_jsb, delay_ackermann, delay_planner,
        rviz_node,
    ])
