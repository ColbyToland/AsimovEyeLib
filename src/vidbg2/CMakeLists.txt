cmake_minimum_required(VERSION 2.8)

project(vidbg2)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(vidbg2 vidbg2.cpp)

target_link_libraries(vidbg2 ${OpenCV_LIBS})
