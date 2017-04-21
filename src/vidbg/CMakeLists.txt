cmake_minimum_required(VERSION 2.8)

project(vidbg)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(vidbg vidbg.cpp)

target_link_libraries(vidbg ${OpenCV_LIBS})
