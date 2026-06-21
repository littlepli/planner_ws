cmake_minimum_required(VERSION 3.8)
project(planner_gazebo_demo)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

add_executable(gazebo_planner_node src/gazebo_planner_node.cpp)
ament_target_dependencies(gazebo_planner_node
  rclcpp nav_msgs visualization_msgs geometry_msgs)

install(TARGETS gazebo_planner_node DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY urdf config worlds launch rviz DESTINATION share/${PROJECT_NAME})

ament_package()