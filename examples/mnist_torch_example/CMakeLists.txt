find_package(Torch REQUIRED)

add_executable(mnist_torch_example)
target_sources(mnist_torch_example PRIVATE mnist_torch_example.cpp)
target_include_directories(mnist_torch_example PRIVATE ${PROJECT_SOURCE_DIR}/dependencies)
target_link_libraries(mnist_torch_example PRIVATE "${TORCH_LIBRARIES}")

if (USE_SIMD)
    target_compile_options(mnist_torch_example PRIVATE ${SIMD_FLAGS})
    target_link_options(mnist_torch_example PRIVATE ${SIMD_FLAGS})
endif ()

configure_file("${PROJECT_SOURCE_DIR}/config.ini" "${CMAKE_CURRENT_BINARY_DIR}/config.ini" COPYONLY)
install(TARGETS mnist_torch_example
        CONFIGURATIONS Debug;Release
        RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR})
