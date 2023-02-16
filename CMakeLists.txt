# cmake -DCMAKE_PREFIX_PATH="/path/to//build_opencv" -DCMAKE_BUILD_TYPE=Release ../..
# cmake -DCMAKE_PREFIX_PATH="/path/to//build_opencv" -DCMAKE_BUILD_TYPE=Debug ../..
# cmake --help-module FindOpenCV
# cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=1 ../..

cmake_minimum_required(VERSION 3.14)
project( MNISTCpp VERSION 0.0.1 )

set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS YES CACHE BOOL "Export all symbols")
option(BUILD_SHARED_LIBS "SHARED or STATIC" OFF)

# project(neuralnet VERSION 0.0.1 DESCRIPTION "custom neural net")
# project(mnist VERSION 0.0.1 DESCRIPTION "mnist parser")

include(GNUInstallDirs)

include_directories( "${CMAKE_SOURCE_DIR}/include;${CMAKE_SOURCE_DIR}/dependencies" )
set(Torch_DIR ${PROJECT_SOURCE_DIR}/dependencies/libtorch/share/cmake/Torch)
find_package(Torch REQUIRED)
set(OpenCV_DIR ${PROJECT_SOURCE_DIR}/dependencies/opencv/lib/cmake/opencv4)
find_package(OpenCV REQUIRED)


add_library(neuralnet ${PROJECT_SOURCE_DIR}/src/NeuralNet/NeuralNet.cpp)
set(NEURALNET_PUBLIC_HEADERS ${PROJECT_SOURCE_DIR}/include/NeuralNet/NeuralNet.hpp)
target_include_directories(neuralnet PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_include_directories(neuralnet PRIVATE ${PROJECT_SOURCE_DIR}/src)
set_target_properties(neuralnet PROPERTIES VERSION ${PROJECT_VERSION})
set_target_properties(neuralnet PROPERTIES SOVERSION ${PROJECT_VERSION})
set_target_properties(neuralnet PROPERTIES PUBLIC_HEADER ${NEURALNET_PUBLIC_HEADERS})


add_library(mnist ${PROJECT_SOURCE_DIR}/src/MNIST/MNIST_Dataset.cpp ${PROJECT_SOURCE_DIR}/src/MNIST/MNIST_Image.cpp)
set(MNIST_PUBLIC_HEADERS 
    ${PROJECT_SOURCE_DIR}/include/MNIST/MNIST_Dataset.hpp
    ${PROJECT_SOURCE_DIR}/include/MNIST/MNIST_Image.hpp
)
target_include_directories(mnist PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_include_directories(mnist PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(mnist ${OpenCV_LIBRARIES})
set_target_properties(mnist PROPERTIES VERSION ${PROJECT_VERSION})
set_target_properties(mnist PROPERTIES SOVERSION ${PROJECT_VERSION})
set_target_properties(mnist PROPERTIES PUBLIC_HEADER "${MNIST_PUBLIC_HEADERS}")

add_executable(mnist_math mnist_math.cpp )
target_link_libraries(mnist_math mnist;neuralnet)
set_target_properties(mnist_math PROPERTIES VERSION ${PROJECT_VERSION})

add_executable(mnist_torch mnist_torch.cpp )
target_link_libraries(mnist_torch "mnist;${TORCH_LIBRARIES}")

add_executable(mnist_example mnist_example.cpp )
target_link_libraries(mnist_example "mnist;${TORCH_LIBRARIES}")

add_executable(mnist_batches mnist_batches.cpp )
target_link_libraries(mnist_batches "mnist;${TORCH_LIBRARIES}")

# configure_file(neuralnet.pc.in neuralnet.pc @ONLY)
# configure_file(mnist.pc.in mnist.pc @ONLY)
# install(TARGETS neuralnet mnist
#     LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
#     ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
#     PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
#     RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
#     )
# install(FILES ${CMAKE_BINARY_DIR}/neuralnet.pc DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/pkgconfig)
# install(FILES ${CMAKE_BINARY_DIR}/mnist.pc DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/pkgconfig)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DDEBUG -D_DEBUG" CACHE STRING "")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")

configure_file("${PROJECT_SOURCE_DIR}/config.ini" "${CMAKE_CURRENT_BINARY_DIR}/config.ini" COPYONLY)

message("")
message("Generated with config types: ${CMAKE_CONFIGURATION_TYPES}")
message("PROJECT_NAME: ${PROJECT_NAME}")
message("PROJECT_VERSION: ${PROJECT_VERSION}")
message("PROJECT_SOURCE_DIR: ${PROJECT_SOURCE_DIR}")
message("CMAKE_SOURCE_DIR: ${CMAKE_SOURCE_DIR}")
message("CMAKE_CXX_FLAGS_DEBUG: ${CMAKE_CXX_FLAGS_DEBUG}")
message("CMAKE_CXX_FLAGS_RELEASE: ${CMAKE_CXX_FLAGS_RELEASE}")
message("BUILD_SHARED_LIBS: ${BUILD_SHARED_LIBS}")
message("CMAKE_INSTALL_PREFIX: ${CMAKE_INSTALL_PREFIX}")
message("CMAKE_INSTALL_LIBDIR: ${CMAKE_INSTALL_LIBDIR}")
message("CMAKE_INSTALL_INCLUDEDIR: ${CMAKE_INSTALL_INCLUDEDIR}")
message("CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")
message("TORCH_CXX_FLAGS: ${TORCH_CXX_FLAGS}")
message("TORCH_LIBRARIES: ${TORCH_LIBRARIES}")
message("TORCH_INCLUDE_DIRS: ${TORCH_INCLUDE_DIRS}")
message("OpenCV_DIR ${OpenCV_DIR}")
message("OpenCV_LIBRARIES ${OpenCV_LIBRARIES}")
message("OpenCV_INCLUDE_DIRS ${OpenCV_INCLUDE_DIRS}")
message("")
