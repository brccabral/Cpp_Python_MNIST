add_executable(mnist_blaze)
target_sources(mnist_blaze PRIVATE mnist_blaze.cpp)
target_include_directories(mnist_blaze PRIVATE ${PROJECT_SOURCE_DIR}/dependencies)
target_link_libraries(mnist_blaze PRIVATE mnist neuralnet_blaze)

configure_file("${PROJECT_SOURCE_DIR}/config.ini" "${CMAKE_CURRENT_BINARY_DIR}/config.ini" COPYONLY)
add_custom_command(
	TARGET mnist_blaze POST_BUILD
	COMMAND ${CMAKE_COMMAND} -E copy
			"${PROJECT_SOURCE_DIR}/config.ini"
			$<TARGET_FILE_DIR:mnist_blaze>)
install(TARGETS mnist_blaze
        CONFIGURATIONS Debug;Release
        RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR})
