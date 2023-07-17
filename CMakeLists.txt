cmake_minimum_required(VERSION 3.10)
project(imgProcess)

# BEFORE COMPILING AND BUILDING ITK, WE NED THE RTK LIBRARY COMPILED AND BUILT WITHIN ITK.
# FOR THAT, GIT CLONE RTK FROM "https://github.com/SimonRit/RTK.git" IN "/.../ITK/MODULES/REMOTE" DIRECTORY.
# THEN CMAKE ITK LIKE SO: "CMAKE .. -DModule_RTK=ON -DITK_USE_GPU=OFF -DRTK_USE_CUDA:BOOL=OFF 
#                          -DCMAKE_INSTALL_PREFIX:PATH=/.../ITK/install"
# THEN DO MAKE AND MAKE INSTALL
find_package(ITK REQUIRED)
include(${ITK_USE_FILE})

add_executable(imgProcess imgProcess.cpp)

target_link_libraries(imgProcess ${ITK_LIBRARIES})
