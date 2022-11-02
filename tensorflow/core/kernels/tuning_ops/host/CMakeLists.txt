cmake_minimum_required(VERSION 3.5)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c11")
set(CMAKE_C_FLAGS_DEBUG "-Wall -g -O0")
set(CMAKE_C_FLAGS_RELEASE "-Wall -O3")

MACRO(HEADER_DIRECTORIES return_list folder)
    message(STATUS "folder=${folder}")
    FILE(GLOB_RECURSE new_list ${folder}/*.h)
    SET(dir_list "")

    FOREACH(file_path ${new_list})
        GET_FILENAME_COMPONENT(dir_path ${file_path} PATH)
        SET(dir_list ${dir_list} ${dir_path})
    ENDFOREACH()

    LIST(REMOVE_DUPLICATES dir_list)
    SET(${return_list} ${dir_list})
ENDMACRO()

HEADER_DIRECTORIES(include_header_dir_list ${CMAKE_CURRENT_SOURCE_DIR}/include)
HEADER_DIRECTORIES(algo_header_dir_list ${CMAKE_CURRENT_SOURCE_DIR}/algorithm)
HEADER_DIRECTORIES(src_header_dir_list ${CMAKE_CURRENT_SOURCE_DIR}/src)
HEADER_DIRECTORIES(suite_header_dir_list ${CMAKE_CURRENT_SOURCE_DIR}/suite)

include_directories(${include_header_dir_list})
include_directories(${algo_header_dir_list})
include_directories(${src_header_dir_list})
include_directories(${suite_header_dir_list})

file(GLOB_RECURSE SRC_ALGO ${CMAKE_CURRENT_SOURCE_DIR}/algorithm/*.c)
file(GLOB_RECURSE SRC_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/*.c)
list(REMOVE_ITEM SRC_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/DataTypes/test/main.c)
file(GLOB_RECURSE SRC_SUITE ${CMAKE_CURRENT_SOURCE_DIR}/suite/*.c)
set(SRC_ALL ${SRC_ALGO} ${SRC_SRC})

project(HOST_RELEASE)
add_library(pta SHARED ${SRC_ALL} ${SRC_SUITE})

install(TARGETS pta
    LIBRARY DESTINATION "./lib/"
)

SET(CPACK_GENERATOR "DEB")
SET(CPACK_DEBIAN_PACKAGE_MAINTAINER "WXS")
INCLUDE(CPack)