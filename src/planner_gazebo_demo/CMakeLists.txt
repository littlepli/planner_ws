cmake_minimum_required(VERSION 3.8)
project(planner_gazebo_demo)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(ackermann_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2 REQUIRED)
find_package(yaml-cpp REQUIRED)

# --- ackermann_teleop ---
add_executable(ackermann_teleop src/ackermann_teleop.cpp)
ament_target_dependencies(ackermann_teleop
  rclcpp ackermann_msgs)

# --- kino_astar_planner ---
add_executable(kino_astar_planner src/kino_astar_planner.cpp)
ament_target_dependencies(kino_astar_planner
  rclcpp nav_msgs geometry_msgs tf2_ros tf2)
target_link_libraries(kino_astar_planner yaml-cpp)

# --- bspline_planner ---
add_executable(bspline_planner src/bspline_planner.cpp)
ament_target_dependencies(bspline_planner
  rclcpp nav_msgs geometry_msgs tf2_ros tf2)
target_link_libraries(bspline_planner yaml-cpp)

# --- ego_planner ---
add_executable(ego_planner src/ego_planner.cpp)
ament_target_dependencies(ego_planner
  rclcpp nav_msgs geometry_msgs tf2_ros tf2)
target_link_libraries(ego_planner yaml-cpp)

install(TARGETS
  ackermann_teleop
  kino_astar_planner
  bspline_planner
  ego_planner
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY urdf config worlds launch rviz maps
  DESTINATION share/${PROJECT_NAME}
  OPTIONAL)

ament_package()