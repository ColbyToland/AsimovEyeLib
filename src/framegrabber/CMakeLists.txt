cmake_minimum_required(VERSION 2.8)

project(framegrabber)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(framegrabber framegrabber.cpp)

target_link_libraries(framegrabber ${OpenCV_LIBS})
