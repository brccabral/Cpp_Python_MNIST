find_package(xtensor REQUIRED)

if (WIN32)
    find_package(xtl REQUIRED)
    add_definitions(-DHAVE_CBLAS=1)
    list(APPEND CMAKE_PREFIX_PATH "${OpenBLAS_DIR}/../../../") # needed for xtensor-blas to find OpenBLAS
    find_package(xtensor-blas REQUIRED)
    find_package(xsimd REQUIRED)
    find_package(TBB REQUIRED)
    set(TBB_PATH "${TBB_DIR}/../../..")
endif ()

add_library(neuralnetxt STATIC)
target_sources(neuralnetxt PRIVATE NeuralNetXT.cpp)
target_include_directories(neuralnetxt
        PUBLIC
        include
        ${PROJECT_SOURCE_DIR}/dependencies/nc
)
target_link_libraries(neuralnetxt PUBLIC xtensor)

if (WIN32)
    target_include_directories(neuralnetxt PUBLIC ${xtl_INCLUDE_DIRS} ${xtensor-blas_INCLUDE_DIRS})
    target_link_libraries(neuralnetxt PUBLIC ${BLAS_LIBRARIES})
endif ()

if (USE_SIMD)
    # find_package(xtensor) tries to find xsim
    target_compile_definitions(neuralnetxt PUBLIC XTENSOR_USE_XSIMD)
    target_compile_options(neuralnetxt PUBLIC ${SIMD_FLAGS_COMPILER})
    target_link_options(neuralnetxt PUBLIC ${SIMD_FLAGS_LINKER})
    target_link_libraries(neuralnetxt PUBLIC xtensor::optimize xsimd)
endif ()

if (USE_TBB)
    # Windows can't compile with both XTENSOR_USE_XSIMD and XTENSOR_USE_TBB
    # https://github.com/xtensor-stack/xtensor/issues/2736
    target_compile_definitions(neuralnetxt PUBLIC XTENSOR_USE_TBB)
    target_link_libraries(neuralnetxt PUBLIC TBB::tbb)
    if (WIN32)
        target_include_directories(neuralnetxt PUBLIC "${TBB_PATH}/include")
        target_link_directories(neuralnetxt PUBLIC "${TBB_PATH}/lib")
    endif (WIN32)
endif ()
