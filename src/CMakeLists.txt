project(Unscented_Kalman_Filter)

cmake_minimum_required (VERSION 3.5)

add_definitions(-std=c++0x)

set(sources
   ./ukf.cpp
   ./main.cpp
   ./tools.cpp)

add_executable(UnscentedKF ${sources})
