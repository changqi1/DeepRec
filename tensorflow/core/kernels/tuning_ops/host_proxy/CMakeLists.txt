cmake_minimum_required(VERSION 3.5)

set(TARGET_NAME "host_proxy")

## to use C/C++11
set (CMAKE_C_STANDARD 11)
set (CMAKE_CXX_STANDARD 11)
set (CMAKE_POSITION_INDEPENDENT_CODE ON)

file(GLOB MAIN_SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
                   ${CMAKE_CURRENT_SOURCE_DIR}/*.c 
                   )

file(GLOB MAIN_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/.h*)

add_library(${TARGET_NAME} SHARED ${MAIN_SRC} ${MAIN_HEADERS})

message(STATUS "HOST_LIB is ${HOST_LIB}")

target_include_directories(${TARGET_NAME} 
PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
PRIVATE
        ${CMAKE_SOURCE_DIR}/host/include
        ${CMAKE_SOURCE_DIR}/host/include/DataTypes
)

target_link_libraries(${TARGET_NAME}
PUBLIC
        ${HOST_LIB}
)
