cmake_minimum_required(VERSION 3.10)
project(imgProcess)

# Find Eigen
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
include_directories(${EIGEN3_INCLUDE_DIR})
#message("Found Eigen3 in: ${EIGEN3_INCLUDE_DIR}")

find_package(ITK REQUIRED)
include(${ITK_USE_FILE})

# YAML
set(YAMLCPP_BINARY_DIR /home/tunok/Work/yaml-cpp/build)
#set(YAML_CONFIG_FILE "${CMAKE_CURRENT_SOURCE_DIR}/exe/Config.yaml")
add_subdirectory(/home/tunok/Work/yaml-cpp ${YAMLCPP_BINARY_DIR})
include_directories(
   /home/tunok/Work/yaml-cpp/include
)

add_executable(imgProcess imgProcess.cpp)

target_link_libraries(imgProcess ${ITK_LIBRARIES} Eigen3::Eigen yaml-cpp)
