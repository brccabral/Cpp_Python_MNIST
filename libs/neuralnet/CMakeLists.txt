find_package(Eigen3 REQUIRED)

add_library(neuralnet STATIC)
target_sources(neuralnet PRIVATE NeuralNet.cpp)
target_include_directories(neuralnet PUBLIC include)
target_link_libraries(neuralnet PUBLIC Eigen3::Eigen)

if (USE_SIMD)
    target_compile_options(neuralnet PUBLIC ${SIMD_FLAGS_COMPILER})
    target_link_options(neuralnet PUBLIC ${SIMD_FLAGS_LINKER})
endif ()
