cmake_minimum_required(VERSION 3.8)
project(planner_gazebo_demo)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(ackermann_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_msgs REQUIRED)
find_package(yaml-cpp REQUIRED)

# --- ackermann_teleop ---
add_executable(ackermann_teleop src/ackermann_teleop.cpp)
ament_target_dependencies(ackermann_teleop
  rclcpp geometry_msgs)

# --- tf_odometry_relay (republishes steering controller's ~/tf_odometry to /tf) ---
add_executable(tf_odometry_relay src/tf_odometry_relay.cpp)
ament_target_dependencies(tf_odometry_relay
  rclcpp tf2_msgs)

# --- auto_explore (hands-free bounded coverage for SLAM mapping) ---
add_executable(auto_explore src/auto_explore.cpp)
ament_target_dependencies(auto_explore
  rclcpp nav_msgs geometry_msgs)

# --- path_player (animate the car along the planned path in RViz) ---
add_executable(path_player src/path_player.cpp)
ament_target_dependencies(path_player
  rclcpp nav_msgs sensor_msgs geometry_msgs tf2_ros)

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
  tf_odometry_relay
  auto_explore
  path_player
  kino_astar_planner
  bspline_planner
  ego_planner
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY urdf config worlds launch rviz maps
  DESTINATION share/${PROJECT_NAME}
  OPTIONAL)

ament_package()