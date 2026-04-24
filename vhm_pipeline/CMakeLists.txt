cmake_minimum_required(VERSION 3.8)
project(vhm_pipeline)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()


# Include headers
include_directories(include)






# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(vhm_interfaces REQUIRED)

add_executable(pipeline_controller_node src/pipeline_controller_node.cpp)

target_include_directories(pipeline_controller_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

ament_target_dependencies(pipeline_controller_node
  rclcpp
  rclcpp_action
  vhm_interfaces
)

install(TARGETS 
  pipeline_controller_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

ament_package()
