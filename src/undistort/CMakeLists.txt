cmake_minimum_required(VERSION 2.8)

project(undistort)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(undistort undistort.cpp)

target_link_libraries(undistort ${OpenCV_LIBS})
