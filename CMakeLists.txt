cmake_minimum_required(VERSION 3.14)
project(MNISTCpp)

include(GNUInstallDirs)

find_package(Eigen3 REQUIRED)
find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)
find_package(xtensor REQUIRED)

option(USE_SIMD "Enables XTENSOR_USE_XSIMD - SIMD optimizations" OFF)
option(USE_TBB "Enables XTENSOR_USE_TBB - parallelism" OFF)
option(NC_USE_MULTITHREAD "Passes NUMCPP_USE_MULTITHREAD to NumCpp" OFF)

add_library(neuralnet
        ${PROJECT_SOURCE_DIR}/src/NeuralNet/NeuralNet.cpp)
if (USE_SIMD)
    target_compile_options(neuralnet PUBLIC "-march=native" "-mtune=native")
endif ()
target_include_directories(neuralnet PUBLIC ${PROJECT_SOURCE_DIR}/include ${EIGEN3_INCLUDE_DIRS})

add_library(neuralnetnc
        ${PROJECT_SOURCE_DIR}/src/NeuralNet/NeuralNetNC.cpp)
if (USE_SIMD)
    target_compile_options(neuralnetnc PUBLIC "-march=native" "-mtune=native")
endif ()
target_include_directories(neuralnetnc PUBLIC ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/dependencies/nc)

add_library(neuralnetxt
        ${PROJECT_SOURCE_DIR}/src/NeuralNet/NeuralNetXT.cpp)
if (USE_SIMD)
    target_compile_definitions(neuralnetxt PUBLIC XTENSOR_USE_XSIMD)
    target_compile_options(neuralnetxt PUBLIC "-march=native" "-mtune=native")
endif ()
if (USE_TBB)
    target_compile_definitions(neuralnetxt PUBLIC XTENSOR_USE_TBB)
    target_link_libraries(neuralnetxt "tbb")
endif ()
target_include_directories(neuralnetxt PUBLIC ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/dependencies/nc)

add_library(mnist
        ${PROJECT_SOURCE_DIR}/src/MNIST/MNIST_Dataset.cpp
        ${PROJECT_SOURCE_DIR}/src/MNIST/MNIST_Image.cpp)
target_link_libraries(mnist ${OpenCV_LIBRARIES})
target_include_directories(mnist PUBLIC ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/dependencies ${PROJECT_SOURCE_DIR}/dependencies/nc ${EIGEN3_INCLUDE_DIRS})

add_executable(mnist_math mnist_math.cpp)
if (USE_SIMD)
    target_compile_options(mnist_math PUBLIC "-march=native" "-mtune=native")
endif ()
target_link_libraries(mnist_math mnist;neuralnet)
install(TARGETS mnist_math
        CONFIGURATIONS Debug;Release
        RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR})

add_executable(mnist_torch mnist_torch.cpp)
if (USE_SIMD)
    target_compile_options(mnist_torch PUBLIC "-march=native" "-mtune=native")
endif ()
target_link_libraries(mnist_torch "mnist;${TORCH_LIBRARIES}")
install(TARGETS mnist_torch
        CONFIGURATIONS Debug;Release
        RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR})

add_executable(mnist_example mnist_example.cpp)
if (USE_SIMD)
    target_compile_options(mnist_example PUBLIC "-march=native" "-mtune=native")
endif ()
target_include_directories(mnist_example PUBLIC ${PROJECT_SOURCE_DIR}/dependencies)
target_link_libraries(mnist_example "${TORCH_LIBRARIES}")
install(TARGETS mnist_example
        CONFIGURATIONS Debug;Release
        RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR})

add_executable(mnist_batches mnist_batches.cpp)
if (USE_SIMD)
    target_compile_options(mnist_batches PUBLIC "-march=native" "-mtune=native")
endif ()
target_include_directories(mnist_batches PUBLIC ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/dependencies)
target_link_libraries(mnist_batches "${TORCH_LIBRARIES}")
install(TARGETS mnist_batches
        CONFIGURATIONS Debug;Release
        RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR})

add_executable(mnist_numcpp mnist_numcpp.cpp)
if (USE_SIMD)
    target_compile_options(mnist_numcpp PUBLIC "-march=native" "-mtune=native")
    target_link_libraries(mnist_numcpp "tbb")
endif ()
if (NC_USE_MULTITHREAD)
    message("NumCpp to use MultiThread")
    target_compile_definitions(mnist_numcpp PUBLIC NUMCPP_USE_MULTITHREAD)
    # if installed `oneTBB`, NUMCPP_USE_MULTITHREAD forces to use `tbb` lib
    target_link_libraries(mnist_numcpp "tbb")
endif ()
target_link_libraries(mnist_numcpp "mnist;neuralnetnc")
install(TARGETS mnist_numcpp
        CONFIGURATIONS Debug;Release
        RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR})

add_executable(mnist_xtensor mnist_xtensor.cpp)
target_include_directories(mnist_xtensor PUBLIC ${xtensor_INCLUDE_DIRS})
target_link_libraries(mnist_xtensor "mnist;neuralnetxt;xtensor")
if (USE_SIMD)
    target_compile_definitions(mnist_xtensor PUBLIC XTENSOR_USE_XSIMD)
    target_compile_options(mnist_xtensor PUBLIC "-march=native" "-mtune=native")
endif ()
if (USE_TBB)
    target_compile_definitions(mnist_xtensor PUBLIC XTENSOR_USE_TBB)
    target_link_libraries(mnist_xtensor "tbb")
endif ()
install(TARGETS mnist_xtensor
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
