cmake_minimum_required(VERSION 3.14)
project(MNISTCpp)

include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 20) # needed for NumCpp

find_package(Eigen3 REQUIRED)
find_package(Torch REQUIRED)
set(OPENCV_MAP_IMPORTED_CONFIG "") # this is needed for compiling with OpenCV in Debug
find_package(OpenCV REQUIRED)
find_package(xtensor REQUIRED)

if(WIN32)
    find_package(xtl REQUIRED)
    add_definitions(-DHAVE_CBLAS=1)
    find_package(OpenBLAS REQUIRED)
    list(APPEND CMAKE_PREFIX_PATH "${OpenBLAS_DIR}/../../../") # needed for xtensor-blas to find OpenBLAS
    find_package(xtensor-blas REQUIRED)
    find_package(xsimd REQUIRED)
    find_package(TBB REQUIRED)
    set(TBB_PATH "${TBB_DIR}/../../..")
endif()

option(USE_SIMD "Enables XTENSOR_USE_XSIMD - SIMD optimizations" OFF)
option(USE_TBB "Enables XTENSOR_USE_TBB - parallelism" OFF)
option(NC_USE_MULTITHREAD "Passes NUMCPP_USE_MULTITHREAD to NumCpp" OFF)
option(NC_NO_USE_BOOST "Passes NUMCPP_NO_USE_BOOST to NumCpp" OFF)

if(WIN32)
    set(SIMD_FLAGS "/arch:AVX2")
else()
    set(SIMD_FLAGS "-march=native" "-mtune=native")
endif(WIN32)

add_library(neuralnet
        ${PROJECT_SOURCE_DIR}/src/NeuralNet/NeuralNet.cpp)
if (USE_SIMD)
    target_compile_options(neuralnet PUBLIC ${SIMD_FLAGS})
endif ()
target_include_directories(neuralnet PUBLIC ${PROJECT_SOURCE_DIR}/include ${EIGEN3_INCLUDE_DIRS})

add_library(neuralnetnc
        ${PROJECT_SOURCE_DIR}/src/NeuralNet/NeuralNetNC.cpp)
if (USE_SIMD)
    target_compile_options(neuralnetnc PUBLIC ${SIMD_FLAGS})
endif ()
if (NC_NO_USE_BOOST)
    target_compile_definitions(neuralnetnc PUBLIC NUMCPP_NO_USE_BOOST)
endif ()
target_include_directories(neuralnetnc PUBLIC ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/dependencies/nc)

add_library(neuralnetxt
        ${PROJECT_SOURCE_DIR}/src/NeuralNet/NeuralNetXT.cpp)
if (WIN32)
    target_include_directories(neuralnetxt PUBLIC ${xtl_INCLUDE_DIRS} ${xtensor-blas_INCLUDE_DIRS})
    target_link_libraries(neuralnetxt PUBLIC ${BLAS_LIBRARIES})
endif ()
if (USE_SIMD)
    target_compile_definitions(neuralnetxt PUBLIC XTENSOR_USE_XSIMD)
    target_compile_options(neuralnetxt PUBLIC ${SIMD_FLAGS})
    target_include_directories(neuralnetxt PUBLIC ${xsimd_INCLUDE_DIRS})
endif ()
if (USE_TBB)
    # Windows can't compile with both XTENSOR_USE_XSIMD and XTENSOR_USE_TBB
    # https://github.com/xtensor-stack/xtensor/issues/2736
    target_compile_definitions(neuralnetxt PUBLIC XTENSOR_USE_TBB)
    target_link_libraries(neuralnetxt PUBLIC TBB::tbb)
    if(WIN32)
        target_include_directories(neuralnetxt PUBLIC "${TBB_PATH}/include")
        target_link_directories(neuralnetxt PUBLIC "${TBB_PATH}/lib")
    endif(WIN32)
endif ()
target_include_directories(neuralnetxt PUBLIC ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/dependencies/nc ${xtensor_INCLUDE_DIRS})

add_library(mnist
        ${PROJECT_SOURCE_DIR}/src/MNIST/MNIST_Dataset.cpp
        ${PROJECT_SOURCE_DIR}/src/MNIST/MNIST_Image.cpp)
target_include_directories(mnist PUBLIC 
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/dependencies
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(mnist PUBLIC ${OpenCV_LIBRARIES})

add_executable(mnist_math mnist_math.cpp)
if (USE_SIMD)
    target_compile_options(mnist_math PUBLIC ${SIMD_FLAGS})
endif ()
target_link_libraries(mnist_math mnist;neuralnet;${OpenCV_LIBRARIES})
install(TARGETS mnist_math
        CONFIGURATIONS Debug;Release
        RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR})

add_executable(mnist_torch mnist_torch.cpp)
if (USE_SIMD)
    target_compile_options(mnist_torch PUBLIC ${SIMD_FLAGS})
endif ()
target_include_directories(mnist_torch PUBLIC ${EIGEN3_INCLUDE_DIRS})
target_link_libraries(mnist_torch "mnist;${TORCH_LIBRARIES}")
install(TARGETS mnist_torch
        CONFIGURATIONS Debug;Release
        RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR})

add_executable(mnist_example mnist_example.cpp)
if (USE_SIMD)
    target_compile_options(mnist_example PUBLIC ${SIMD_FLAGS})
endif ()
target_include_directories(mnist_example PUBLIC ${PROJECT_SOURCE_DIR}/dependencies)
target_link_libraries(mnist_example "${TORCH_LIBRARIES}")
install(TARGETS mnist_example
        CONFIGURATIONS Debug;Release
        RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR})

add_executable(mnist_batches mnist_batches.cpp)
if (USE_SIMD)
    target_compile_options(mnist_batches PUBLIC ${SIMD_FLAGS})
endif ()
target_include_directories(mnist_batches PUBLIC ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/dependencies)
target_link_libraries(mnist_batches "${TORCH_LIBRARIES}")
install(TARGETS mnist_batches
        CONFIGURATIONS Debug;Release
        RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR})

add_executable(mnist_numcpp mnist_numcpp.cpp)
if (USE_SIMD)
    target_compile_options(mnist_numcpp PUBLIC ${SIMD_FLAGS})
    target_link_libraries(mnist_numcpp TBB::tbb)
    if(WIN32)
        target_link_directories(mnist_numcpp PUBLIC "${TBB_PATH}/lib")
    endif(WIN32)
endif ()
if (NC_USE_MULTITHREAD)
    message("NumCpp to use MultiThread")
    target_compile_definitions(mnist_numcpp PUBLIC NUMCPP_USE_MULTITHREAD)
    # if installed `oneTBB`, NUMCPP_USE_MULTITHREAD forces to use `tbb` lib
    target_link_libraries(mnist_numcpp TBB::tbb)
    if(WIN32)
        target_link_directories(mnist_numcpp PUBLIC "${TBB_PATH}/lib")
    endif(WIN32)
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
    target_compile_options(mnist_xtensor PUBLIC ${SIMD_FLAGS})
    target_include_directories(mnist_xtensor PUBLIC ${xsimd_INCLUDE_DIRS})
endif ()
if (USE_TBB)
    # Windows can't compile with both XTENSOR_USE_XSIMD and XTENSOR_USE_TBB
    # https://github.com/xtensor-stack/xtensor/issues/2736
    target_compile_definitions(mnist_xtensor PUBLIC XTENSOR_USE_TBB)
    target_link_libraries(mnist_xtensor TBB::tbb)
    if(WIN32)
        target_include_directories(mnist_xtensor PUBLIC "${TBB_PATH}/include")
        target_link_directories(mnist_xtensor PUBLIC "${TBB_PATH}/lib")
    endif(WIN32)
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
message("OpenCV_VERSION ${OpenCV_VERSION}")
message("OpenCV_LIBRARIES: ${OpenCV_LIBRARIES}")
message("OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")
message("EIGEN3_INCLUDE_DIRS: ${EIGEN3_INCLUDE_DIRS}")
if (WIN32)
message("xtensor_INCLUDE_DIRS ${xtensor_INCLUDE_DIRS}")
message("xtl_INCLUDE_DIRS ${xtl_INCLUDE_DIRS}")
message("OpenBLAS_INCLUDE_DIRS ${OpenBLAS_INCLUDE_DIRS}")
message("xtensor-blas_INCLUDE_DIRS ${xtensor-blas_INCLUDE_DIRS}")
message("BLAS_LIBRARIES ${BLAS_LIBRARIES}")
message("xsimd_INCLUDE_DIRS ${xsimd_INCLUDE_DIRS}")
message("TBB_FOUND ${TBB_FOUND}")
message("TBB_DIR ${TBB_DIR}")
message("TBB_PATH ${TBB_PATH}")
endif ()
message("")
