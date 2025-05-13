find_package(Eigen3 REQUIRED)

add_executable(mnist_torch)
target_sources(mnist_torch PRIVATE mnist_torch.cpp)
target_include_directories(mnist_torch PRIVATE ${PROJECT_SOURCE_DIR}/dependencies)
target_link_libraries(mnist_torch PRIVATE Eigen3::Eigen mnist torchnet)

configure_file("${PROJECT_SOURCE_DIR}/config.ini" "${CMAKE_CURRENT_BINARY_DIR}/config.ini" COPYONLY)
install(TARGETS mnist_torch
        CONFIGURATIONS Debug;Release
        RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR})
