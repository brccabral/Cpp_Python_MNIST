if (EXISTS "${PROJECT_SOURCE_DIR}/.venv")
    set(Python3_ROOT_DIR "${PROJECT_SOURCE_DIR}/.venv")
endif ()

if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.venv")
    set(Python3_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/.venv")
endif ()

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(Python3 COMPONENTS NumPy REQUIRED)

add_library(neuralnet_cnumpy STATIC)
target_sources(neuralnet_cnumpy PRIVATE NeuralNet_CNumpy.cpp)
target_include_directories(neuralnet_cnumpy PUBLIC include)
target_link_libraries(neuralnet_cnumpy PUBLIC Python3::Python Python3::Module Python3::NumPy)

if (USE_SIMD)
    target_compile_options(neuralnet_cnumpy PUBLIC ${SIMD_FLAGS})
    target_link_options(neuralnet_cnumpy PUBLIC ${SIMD_FLAGS})
endif ()
