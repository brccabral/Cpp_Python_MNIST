add_executable(mnist_numcpp)
target_sources(mnist_numcpp PRIVATE mnist_numcpp.cpp)
target_include_directories(mnist_numcpp PRIVATE ${PROJECT_SOURCE_DIR}/dependencies)
target_link_libraries(mnist_numcpp PRIVATE mnist neuralnetnc)

configure_file("${PROJECT_SOURCE_DIR}/config.ini" "${CMAKE_CURRENT_BINARY_DIR}/config.ini" COPYONLY)
install(TARGETS mnist_numcpp
        CONFIGURATIONS Debug;Release
        RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR})
