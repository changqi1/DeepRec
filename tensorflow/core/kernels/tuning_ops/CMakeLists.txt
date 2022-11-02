cmake_minimum_required(VERSION 3.5)
project(host_tuning_matmul VERSION 0.1.0)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
    message(STATUS "CMAKE_BUILD_TYPE is undefined. Set default build type ${CMAKE_BUILD_TYPE}.")
endif()

set(CMAKE_CXX_FLAGS_DEBUG "-Wall -g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-Wall -O3")

find_library(HOST_LIB
    NAMES pta
)

add_subdirectory(host_proxy)
add_subdirectory(host_proxy_test)
