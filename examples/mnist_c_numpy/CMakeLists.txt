add_executable(mnist_c_numpy)
target_sources(mnist_c_numpy PRIVATE mnist_c_numpy.cpp)
target_include_directories(mnist_c_numpy PRIVATE ${PROJECT_SOURCE_DIR}/dependencies)
target_link_libraries(mnist_c_numpy PRIVATE mnist neuralnet_cnumpy)

configure_file("${PROJECT_SOURCE_DIR}/config.ini" "${CMAKE_CURRENT_BINARY_DIR}/config.ini" COPYONLY)
install(TARGETS mnist_c_numpy
        CONFIGURATIONS Debug;Release
        RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR})
