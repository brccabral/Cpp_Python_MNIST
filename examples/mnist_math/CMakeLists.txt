add_executable(mnist_math)
target_sources(mnist_math PRIVATE mnist_math.cpp)
target_include_directories(mnist_math PRIVATE ${PROJECT_SOURCE_DIR}/dependencies)
target_link_libraries(mnist_math PRIVATE mnist neuralnet)

configure_file("${PROJECT_SOURCE_DIR}/config.ini" "${CMAKE_CURRENT_BINARY_DIR}/config.ini" COPYONLY)
add_custom_command(
	TARGET mnist_math POST_BUILD
	COMMAND ${CMAKE_COMMAND} -E copy
			"${PROJECT_SOURCE_DIR}/config.ini"
			$<TARGET_FILE_DIR:mnist_math>)
install(TARGETS mnist_math
        CONFIGURATIONS Debug;Release
        RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR})
