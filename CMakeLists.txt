cmake_minimum_required(VERSION 3.10)
project(TSP_GA)

set(CMAKE_CXX_STANDARD 20)

include_directories(src)

add_executable(tsp_ga
    src/main.cpp
    src/tsp_ga.cpp
)