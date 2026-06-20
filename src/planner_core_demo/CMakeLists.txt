cmake_minimum_required(VERSION 3.8)
project(planner_core_demo)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(std_msgs REQUIRED)

set(PLANNER_DEMO_DEPS
  rclcpp
  visualization_msgs
  geometry_msgs
  nav_msgs
  std_msgs
)

set(PLANNER_DEMO_TARGETS
  random_map_demo
  astar_grid_demo
  kino_astar_demo
  bspline_backend_demo
  ego_planner_demo
  closed_loop_demo
  planner_benchmark
  mapping_demo
  ackermann_kino_astar_demo
)

foreach(target ${PLANNER_DEMO_TARGETS})
  add_executable(${target} src/${target}.cpp)
  ament_target_dependencies(${target} ${PLANNER_DEMO_DEPS})
endforeach()

install(TARGETS
  ${PLANNER_DEMO_TARGETS}
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  rviz
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
