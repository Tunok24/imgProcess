cmake_minimum_required(VERSION 3.10)
project(extractSlices) # imgProcess # extractSlices

# Find Eigen
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
include_directories(${EIGEN3_INCLUDE_DIR})

find_package(ITK REQUIRED)
include(${ITK_USE_FILE})

# YAML
set(YAMLCPP_BINARY_DIR /home/tunok/Work/yaml-cpp/build)
add_subdirectory(/home/tunok/Work/yaml-cpp ${YAMLCPP_BINARY_DIR})
include_directories(/home/tunok/Work/yaml-cpp/include)

add_executable(extractSlices extractSlices.cpp)

target_link_libraries(extractSlices ${ITK_LIBRARIES} Eigen3::Eigen yaml-cpp)
