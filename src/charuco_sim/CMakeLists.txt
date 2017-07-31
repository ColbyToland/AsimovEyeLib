cmake_minimum_required(VERSION 2.8)

# Get project name from top directory
string(REPLACE "/" ";" DIR_LIST ${CMAKE_CURRENT_SOURCE_DIR})
list(REVERSE DIR_LIST)
list(GET DIR_LIST 0 PROJ_NAME)

project(${PROJ_NAME})

find_package(OpenCV REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${OpenGL_INCLUDE_DIRS})
include_directories(${GLUT_INCLUDE_DIRS})

add_executable(${PROJ_NAME} ${PROJ_NAME}.cpp)

target_link_libraries(${PROJ_NAME} ${OpenCV_LIBS} ${OpenGL_LIBS} ${GLUT_LIBRARY})
