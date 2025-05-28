find_package(OpenBLAS REQUIRED)

add_library(neuralnetopenblas STATIC)
target_sources(neuralnetopenblas PRIVATE NeuralNetOpenBLAS.cpp)
target_include_directories(neuralnetopenblas
        PUBLIC
        include
        ${OpenBLAS_INCLUDE_DIRS}
        ${PROJECT_SOURCE_DIR}/include
)
if (WIN32)
    target_link_libraries(neuralnetopenblas PUBLIC OpenBLAS::OpenBLAS)
else ()
    target_link_libraries(neuralnetopenblas PUBLIC ${OpenBLAS_LIBRARIES})
endif ()

# neuralnetopenblas requires SIMD flags
target_compile_options(neuralnetopenblas PUBLIC ${SIMD_FLAGS_COMPILER})
target_link_options(neuralnetopenblas PUBLIC ${SIMD_FLAGS_LINKER})
