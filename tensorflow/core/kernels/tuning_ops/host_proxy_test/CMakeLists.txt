cmake_minimum_required(VERSION 3.5)

set(TARGET_NAME "host_proxy_test")

set(CMAKE_CXX_FLAGS "-std=c++17 -fopenmp -mavx512f -mfma -fPIC -Wno-unknown-pragmas")

add_executable(${TARGET_NAME} host_proxy_test.cpp)

target_include_directories(${TARGET_NAME} 
PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
PRIVATE
        ${CMAKE_SOURCE_DIR}/host_proxy
)

target_link_libraries(${TARGET_NAME}
PRIVATE
        host_proxy
)
