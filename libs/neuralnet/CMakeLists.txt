find_package(Eigen3 REQUIRED)

add_library(neuralnet STATIC)
target_sources(neuralnet PRIVATE NeuralNet.cpp)
target_include_directories(neuralnet PUBLIC include ${EIGEN3_INCLUDE_DIRS})

if (USE_SIMD)
    target_compile_options(neuralnet PUBLIC ${SIMD_FLAGS})
    target_link_options(neuralnet PUBLIC ${SIMD_FLAGS})
endif ()
