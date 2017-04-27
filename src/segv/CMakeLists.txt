cmake_minimum_required(VERSION 2.8)

project(segv)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(segv segv.cpp)

target_link_libraries(segv ${OpenCV_LIBS})
