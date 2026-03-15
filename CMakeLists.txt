cmake_minimum_required(VERSION 3.23)
project(MNISTCpp)

include(GNUInstallDirs)

# https://xtensor.readthedocs.io/en/latest/build-options.html#external-dependencies
option(USE_SIMD "Enables XTENSOR_USE_XSIMD - SIMD optimizations" OFF)
option(USE_TBB "Enables XTENSOR_USE_TBB - parallelism" OFF)
option(NC_USE_MULTITHREAD "Passes NUMCPP_USE_MULTITHREAD to NumCpp" OFF)
option(NC_NO_USE_BOOST "Passes NUMCPP_NO_USE_BOOST to NumCpp" OFF)
option(SAVE_IMAGES "Use OpenCV to save images" OFF)

# Windows can't compile with both XTENSOR_USE_XSIMD and XTENSOR_USE_TBB
if (USE_SIMD)
    set(XTENSOR_USE_XSIMD ON CACHE BOOL "")
endif ()
if (USE_TBB)
    set(XTENSOR_USE_TBB ON CACHE BOOL "")
endif ()

include(CheckCXXCompilerFlag)
if (WIN32)
check_cxx_compiler_flag("/openmp" COMPILER_SUPPORT_OMP)
else ()
check_cxx_compiler_flag("-fopenmp" COMPILER_SUPPORT_OMP)
endif ()
message("COMPILER_SUPPORT_OMP ${COMPILER_SUPPORT_OMP}")

if (WIN32)
    set(SIMD_FLAGS_COMPILER "/arch:AVX2" "/fp:fast" "/GL" CACHE STRING "")
    set(SIMD_FLAGS_LINKER "/LTCG" CACHE STRING "")
    if (COMPILER_SUPPORT_OMP)
        set(SIMD_FLAGS_COMPILER ${SIMD_FLAGS_COMPILER} "/openmp" CACHE STRING "" FORCE)
        set(SIMD_FLAGS_LINKER ${SIMD_FLAGS_LINKER} "/openmp" CACHE STRING "" FORCE)
    endif ()
else ()
    set(SIMD_FLAGS_COMPILER "-march=native" "-mtune=native" "-ffast-math" CACHE STRING "")
    if (COMPILER_SUPPORT_OMP)
        set(SIMD_FLAGS_COMPILER ${SIMD_FLAGS_COMPILER} "-fopenmp" CACHE STRING "" FORCE)
        set(SIMD_FLAGS_LINKER "-fopenmp" CACHE STRING "" FORCE)
    endif ()
endif (WIN32)

if (NC_NO_USE_BOOST)
    set(NUMCPP_NO_USE_BOOST ON CACHE BOOL "")
endif ()

install(FILES config.ini
        DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR})

# SimpleIni: common library for all examples
find_package(SimpleIni REQUIRED)

add_subdirectory(libs)
add_subdirectory(examples)
