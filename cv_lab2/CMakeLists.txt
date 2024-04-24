cmake_minimum_required(VERSION 3.10)
PROJECT(conv)

set (CMAKE_CXX_STANDARD 17)

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/build)

add_library(conv SHARED ${CMAKE_CURRENT_SOURCE_DIR}/src/conv.cpp)
target_include_directories(conv PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(main test.cpp)                         
target_link_libraries(main PRIVATE conv)     