add_executable(mnist_torch_batches)
target_sources(mnist_torch_batches PRIVATE mnist_torch_batches.cpp)
target_include_directories(mnist_torch_batches PRIVATE ${PROJECT_SOURCE_DIR}/dependencies)
target_link_libraries(mnist_torch_batches PRIVATE torchnet)

configure_file("${PROJECT_SOURCE_DIR}/config.ini" "${CMAKE_CURRENT_BINARY_DIR}/config.ini" COPYONLY)
add_custom_command(
	TARGET mnist_torch_batches POST_BUILD
	COMMAND ${CMAKE_COMMAND} -E copy
			"${PROJECT_SOURCE_DIR}/config.ini"
			$<TARGET_FILE_DIR:mnist_torch_batches>)
install(TARGETS mnist_torch_batches
        CONFIGURATIONS Debug;Release
        RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR})
