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
    set(XTENSOR_USE_XSIMD ON)
endif ()
if (USE_TBB)
    set(XTENSOR_USE_TBB ON)
endif ()

include(CheckCXXCompilerFlag)
if (WIN32)
check_cxx_compiler_flag("/openmp" COMPILER_SUPPORT_OMP)
else ()
check_cxx_compiler_flag("-fopenmp" COMPILER_SUPPORT_OMP)
endif ()
message("COMPILER_SUPPORT_OMP ${COMPILER_SUPPORT_OMP}")

if (WIN32)
    set(SIMD_FLAGS_COMPILER /arch:AVX2 /fp:fast /GL)
    set(SIMD_FLAGS_LINKER /LTCG)
else ()
    set(SIMD_FLAGS_COMPILER "-march=native" "-mtune=native" "-fopenmp")
    set(SIMD_FLAGS_LINKER "-fopenmp")
endif (WIN32)

install(FILES config.ini
        DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR})

add_subdirectory(libs)
add_subdirectory(examples)
