# cmake -DCMAKE_PREFIX_PATH="/path/to//build_opencv" -DCMAKE_BUILD_TYPE=Release ..
# cmake -DCMAKE_PREFIX_PATH="/path/to//build_opencv" -DCMAKE_BUILD_TYPE=Debug ..

cmake_minimum_required(VERSION 3.14)
project( MNISTCpp )

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -DDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")

find_package(OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories( "./include" )
add_executable(converter_mnist converter_mnist.cpp )
target_link_libraries(converter_mnist ${OpenCV_LIBS})

message("")
message("Generated with config types: ${CMAKE_CONFIGURATION_TYPES}")
message("CMAKE_CXX_FLAGS_DEBUG: ${CMAKE_CXX_FLAGS_DEBUG}")
message("CMAKE_CXX_FLAGS_RELEASE: ${CMAKE_CXX_FLAGS_RELEASE}")
message("")
