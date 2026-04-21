cmake_minimum_required(VERSION 3.8)
project(vhm_pipeline)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)

add_executable(pipeline_controller_node src/pipeline_controller_node.cpp)
ament_target_dependencies(pipeline_controller_node
rclcpp)

install(TARGETS 
  pipeline_controller_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
