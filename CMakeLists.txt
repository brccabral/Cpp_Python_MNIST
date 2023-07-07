# cmake -DCMAKE_PREFIX_PATH="/path/to//build_opencv" -DCMAKE_BUILD_TYPE=Release ../..
# cmake -DCMAKE_PREFIX_PATH="/path/to//build_opencv" -DCMAKE_BUILD_TYPE=Debug ../..
# cmake --help-module FindOpenCV
# cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=1 ../..

cmake_minimum_required(VERSION 3.14)
project( MNISTCpp )

include(GNUInstallDirs)

find_package(Eigen3 REQUIRED)
find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/dependencies ${EIGEN3_INCLUDE_DIRS})

add_library(neuralnet STATIC 
        ${PROJECT_SOURCE_DIR}/src/NeuralNet/NeuralNet.cpp)
target_include_directories(neuralnet PUBLIC ${PROJECT_SOURCE_DIR}/include)

add_library(mnist STATIC 
        ${PROJECT_SOURCE_DIR}/src/MNIST/MNIST_Dataset.cpp 
        ${PROJECT_SOURCE_DIR}/src/MNIST/MNIST_Image.cpp)
target_link_libraries(mnist ${OpenCV_LIBRARIES})
target_include_directories(mnist PUBLIC ${PROJECT_SOURCE_DIR}/include)

add_executable(mnist_math mnist_math.cpp )
target_link_libraries(mnist_math mnist;neuralnet)
install(TARGETS mnist_math
        CONFIGURATIONS Debug;Release
        RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR})

add_executable(mnist_torch mnist_torch.cpp )
target_link_libraries(mnist_torch "mnist;${TORCH_LIBRARIES}")
install(TARGETS mnist_torch
        CONFIGURATIONS Debug;Release
        RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR})

add_executable(mnist_example mnist_example.cpp )
target_link_libraries(mnist_example "${TORCH_LIBRARIES}")
install(TARGETS mnist_example
        CONFIGURATIONS Debug;Release
        RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR})

add_executable(mnist_batches mnist_batches.cpp )
target_link_libraries(mnist_batches "${TORCH_LIBRARIES}")
install(TARGETS mnist_batches
        CONFIGURATIONS Debug;Release
        RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR})

configure_file("${PROJECT_SOURCE_DIR}/config.ini" "${CMAKE_CURRENT_BINARY_DIR}/config.ini" COPYONLY)
install(FILES config.ini
        DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR})

message("")
message("Generated with config types: ${CMAKE_CONFIGURATION_TYPES}")
message("PROJECT_NAME: ${PROJECT_NAME}")
message("PROJECT_SOURCE_DIR: ${PROJECT_SOURCE_DIR}")
message("CMAKE_CURRENT_BINARY_DIR: ${CMAKE_CURRENT_BINARY_DIR}")
message("CMAKE_SOURCE_DIR: ${CMAKE_SOURCE_DIR}")
message("CMAKE_CXX_FLAGS_DEBUG: ${CMAKE_CXX_FLAGS_DEBUG}")
message("CMAKE_CXX_FLAGS_RELEASE: ${CMAKE_CXX_FLAGS_RELEASE}")
message("CMAKE_INSTALL_PREFIX: ${CMAKE_INSTALL_PREFIX}")
message("CMAKE_INSTALL_LIBDIR: ${CMAKE_INSTALL_LIBDIR}")
message("CMAKE_INSTALL_BINDIR: ${CMAKE_INSTALL_BINDIR}")
message("CMAKE_INSTALL_INCLUDEDIR: ${CMAKE_INSTALL_INCLUDEDIR}")
message("TORCH_CXX_FLAGS: ${TORCH_CXX_FLAGS}")
message("TORCH_LIBRARIES: ${TORCH_LIBRARIES}")
message("TORCH_INCLUDE_DIRS: ${TORCH_INCLUDE_DIRS}")
message("OpenCV_DIR: ${OpenCV_DIR}")
message("OpenCV_LIBRARIES: ${OpenCV_LIBRARIES}")
message("OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")
message("EIGEN3_INCLUDE_DIRS: ${EIGEN3_INCLUDE_DIRS}")
message("")
