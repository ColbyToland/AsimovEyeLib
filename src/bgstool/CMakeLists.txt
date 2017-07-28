cmake_minimum_required(VERSION 2.8)

# Get project name from top directory
string(REPLACE "/" ";" DIR_LIST ${CMAKE_CURRENT_SOURCE_DIR})
list(REVERSE DIR_LIST)
list(GET DIR_LIST 0 PROJ_NAME)

project(${PROJ_NAME})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")

include_directories(../../libs)

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Background Subtraction Library
file(GLOB_RECURSE bgs_src ../../libs/package_bgs/*.cpp ../../libs/package_bgs/*.c)
file(GLOB_RECURSE analysis_src ../../libs/package_analysis/*.cpp)

# Build
add_executable(${PROJ_NAME} ${PROJ_NAME}.cpp ${bgs_src} ${analysis_src})
target_link_libraries(${PROJ_NAME} ${OpenCV_LIBS})
