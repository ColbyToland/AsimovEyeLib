cmake_minimum_required(VERSION 2.8)

project(bgsubtract)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(bgsubtract bgsubtract.cpp)

target_link_libraries(bgsubtract ${OpenCV_LIBS})
