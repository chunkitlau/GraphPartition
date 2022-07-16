# the whole CMakeLists.txt for project graphPartition
CMAKE_MINIMUM_REQUIRED(VERSION 3.16)

PROJECT(graphPartition VERSION 1.0)

set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_STANDARD 11)

SET(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-pthread")

add_library(graphPartition SHARED
  src/partition.cpp
  include/partition.hpp

  src/graph.cpp
  include/graph.hpp

  src/utils.cpp
  include/utils.hpp
)

target_include_directories(graphPartition PUBLIC
  include
)

add_executable(partition
  src/partition.cpp
)

target_link_libraries(partition graphPartition)
