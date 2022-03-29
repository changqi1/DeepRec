cmake_minimum_required(VERSION 3.5)

project(PTA)
include_directories(include)
include_directories(include/DataTypes)
include_directories(suite/Ackley)
include_directories(suite/ExpSinFunc)
include_directories(suite/NetworkAI)
link_directories(.)
link_libraries(pta)

add_executable(pta_launcher suite/pta.c)
target_link_libraries(pta_launcher m libpta.so)

