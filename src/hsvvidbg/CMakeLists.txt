cmake_minimum_required(VERSION 2.8)

project(hsvvidbg)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(hsvvidbg hsvvidbg.cpp)

target_link_libraries(hsvvidbg ${OpenCV_LIBS})
