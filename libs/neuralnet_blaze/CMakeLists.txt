find_package(blaze)

add_library(neuralnet_blaze STATIC)
target_sources(neuralnet_blaze PRIVATE NeuralNet_Blaze.cpp)
target_include_directories(neuralnet_blaze PUBLIC include)
target_link_libraries(neuralnet_blaze PUBLIC blaze::blaze)

if (USE_SIMD)
    target_compile_options(neuralnet_blaze PUBLIC ${SIMD_FLAGS})
    target_link_options(neuralnet_blaze PUBLIC ${SIMD_FLAGS})
endif ()
