cmake_minimum_required(VERSION 3.20)
project(STEP_Speciale_Marc)

set(CMAKE_CXX_STANDARD 14)
add_executable(STEP_Speciale_Marc
        ccd.cpp
        ccd.hpp
        )
find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})
target_link_libraries(STEP_Speciale_Marc ${PYTHON_LIBRARIES})