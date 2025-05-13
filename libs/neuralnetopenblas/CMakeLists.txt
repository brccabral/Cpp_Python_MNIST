find_package(OpenBLAS REQUIRED)

add_library(neuralnetopenblas STATIC)
target_sources(neuralnetopenblas PRIVATE NeuralNetOpenBLAS.cpp)
target_include_directories(neuralnetopenblas
        PUBLIC
        include
        ${OpenBLAS_INCLUDE_DIRS}
        ${PROJECT_SOURCE_DIR}/include
)
target_link_libraries(neuralnetopenblas PUBLIC ${OpenBLAS_LIBRARIES})

if (USE_SIMD)
    target_compile_options(neuralnetopenblas PUBLIC ${SIMD_FLAGS})
    target_link_options(neuralnetopenblas PUBLIC ${SIMD_FLAGS})
endif ()
