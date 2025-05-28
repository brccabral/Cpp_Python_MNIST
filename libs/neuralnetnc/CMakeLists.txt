add_library(neuralnetnc STATIC)
target_sources(neuralnetnc PRIVATE NeuralNetNC.cpp)
target_include_directories(neuralnetnc PUBLIC include ${PROJECT_SOURCE_DIR}/dependencies/nc)

target_compile_features(neuralnetnc PUBLIC cxx_std_20)

if (USE_SIMD)
    find_package(TBB REQUIRED)

    target_compile_options(neuralnetnc PUBLIC ${SIMD_FLAGS_COMPILER})
    target_link_options(neuralnetnc PUBLIC ${SIMD_FLAGS_LINKER})
    target_link_libraries(neuralnetnc PUBLIC TBB::tbb)

    if (WIN32)
        target_link_directories(neuralnetnc PUBLIC "${TBB_PATH}/lib")
    endif (WIN32)
endif ()

if (NC_NO_USE_BOOST)
    target_compile_definitions(neuralnetnc PUBLIC NUMCPP_NO_USE_BOOST)
endif ()

if (NC_USE_MULTITHREAD)
    message("NumCpp to use MultiThread")
    find_package(TBB REQUIRED)
    target_compile_definitions(neuralnetnc PUBLIC NUMCPP_USE_MULTITHREAD)
    # if installed `oneTBB`, NUMCPP_USE_MULTITHREAD forces to use `tbb` lib
    target_link_libraries(neuralnetnc PUBLIC TBB::tbb)
    if (WIN32)
        target_link_directories(neuralnetnc PUBLIC "${TBB_PATH}/lib")
    endif (WIN32)
endif ()

