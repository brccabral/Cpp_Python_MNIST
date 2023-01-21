cmake_minimum_required(VERSION 3.14)
project( MNISTCpp )

find_package(OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable(converter_mnist converter_mnist.cpp )
target_link_libraries(converter_mnist ${OpenCV_LIBS})
