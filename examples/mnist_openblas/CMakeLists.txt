add_executable(mnist_openblas)
target_sources(mnist_openblas PRIVATE mnist_openblas.cpp)
target_include_directories(mnist_openblas PRIVATE ${PROJECT_SOURCE_DIR}/dependencies)
target_link_libraries(mnist_openblas PRIVATE mnist neuralnetopenblas)

configure_file("${PROJECT_SOURCE_DIR}/config.ini" "${CMAKE_CURRENT_BINARY_DIR}/config.ini" COPYONLY)
install(TARGETS mnist_openblas
        CONFIGURATIONS Debug;Release
        RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR})
