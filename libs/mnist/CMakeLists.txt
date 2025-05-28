add_library(mnist STATIC)
target_sources(mnist PRIVATE MNIST_Dataset.cpp MNIST_Image.cpp)
target_include_directories(mnist PUBLIC include)

if (USE_SIMD)
    target_compile_options(mnist PUBLIC ${SIMD_FLAGS_COMPILER})
    target_link_options(mnist PUBLIC ${SIMD_FLAGS_LINKER})
endif ()

if (SAVE_IMAGES)
    set(OPENCV_MAP_IMPORTED_CONFIG "") # this is needed for compiling with OpenCV in Debug
    find_package(OpenCV REQUIRED)

    target_include_directories(mnist PUBLIC ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(mnist PUBLIC ${OpenCV_LIBRARIES})
    target_compile_definitions(mnist PUBLIC CV_SAVE_IMAGES)
endif (SAVE_IMAGES)
