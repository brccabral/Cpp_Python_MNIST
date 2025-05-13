add_executable(mnist_math)
target_sources(mnist_math PRIVATE mnist_math.cpp)
target_link_libraries(mnist_math PRIVATE mnist neuralnet)

configure_file("${PROJECT_SOURCE_DIR}/config.ini" "${CMAKE_CURRENT_BINARY_DIR}/config.ini" COPYONLY)
install(TARGETS mnist_math
        CONFIGURATIONS Debug;Release
        RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR})
