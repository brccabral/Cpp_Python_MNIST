add_executable(mnist_xtensor)
target_sources(mnist_xtensor PRIVATE mnist_xtensor.cpp)
target_include_directories(mnist_xtensor PRIVATE ${PROJECT_SOURCE_DIR}/dependencies)
target_link_libraries(mnist_xtensor PRIVATE mnist neuralnetxt)

configure_file("${PROJECT_SOURCE_DIR}/config.ini" "${CMAKE_CURRENT_BINARY_DIR}/config.ini" COPYONLY)
install(TARGETS mnist_xtensor
        CONFIGURATIONS Debug;Release
        RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR})
