cmake_minimum_required(VERSION 3.10)

project(imgProcess)

find_package(ITK REQUIRED)
include(${ITK_USE_FILE})

add_executable(imgProcess imgProcess.cpp)

target_link_libraries(imgProcess ${ITK_LIBRARIES})
