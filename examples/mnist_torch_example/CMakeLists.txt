find_package(Torch REQUIRED)

add_executable(mnist_torch_example)
target_sources(mnist_torch_example PRIVATE mnist_torch_example.cpp)
target_include_directories(mnist_torch_example PRIVATE ${PROJECT_SOURCE_DIR}/dependencies)
target_link_libraries(mnist_torch_example PRIVATE "${TORCH_LIBRARIES}")

if (USE_SIMD)
    target_compile_options(mnist_torch_example PRIVATE ${SIMD_FLAGS_COMPILER})
    target_link_options(mnist_torch_example PRIVATE ${SIMD_FLAGS_LINKER})
endif ()

configure_file("${PROJECT_SOURCE_DIR}/config.ini" "${CMAKE_CURRENT_BINARY_DIR}/config.ini" COPYONLY)
add_custom_command(
	TARGET mnist_torch_example POST_BUILD
	COMMAND ${CMAKE_COMMAND} -E copy
			"${PROJECT_SOURCE_DIR}/config.ini"
			$<TARGET_FILE_DIR:mnist_torch_example>)
install(TARGETS mnist_torch_example
        CONFIGURATIONS Debug;Release
        RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR})
