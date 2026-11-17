cmake_minimum_required(VERSION 3.20)
project(gridbot_interfaces)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(sensor_msgs REQUIRED)

set(msg_files
  "msg/SimpleAruco.msg"
  "msg/RouteNode.msg"
  "msg/RouteRunMetadata.msg"
  "msg/RouteStepResult.msg"
  "msg/StampedRoute.msg"
)

set(srv_files
  "srv/GenerateMap.srv"
  "srv/GenerateRoute.srv"
  "srv/GetDims.srv"
  "srv/AngleCamera.srv"
)

rosidl_generate_interfaces(${PROJECT_NAME}
  ${msg_files}
  ${srv_files}
  DEPENDENCIES builtin_interfaces sensor_msgs
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
