cmake_minimum_required(VERSION 3.23)
project(MNISTCpp)

include(GNUInstallDirs)

# https://xtensor.readthedocs.io/en/latest/build-options.html#external-dependencies
option(USE_SIMD "Enables XTENSOR_USE_XSIMD - SIMD optimizations" OFF)
option(USE_TBB "Enables XTENSOR_USE_TBB - parallelism" OFF)
option(NC_USE_MULTITHREAD "Passes NUMCPP_USE_MULTITHREAD to NumCpp" OFF)
option(NC_NO_USE_BOOST "Passes NUMCPP_NO_USE_BOOST to NumCpp" OFF)
option(SAVE_IMAGES "Use OpenCV to save images" OFF)

if (USE_SIMD)
    set(XTENSOR_USE_XSIMD ON)
endif ()
if (USE_TBB)
    set(XTENSOR_USE_TBB ON)
endif ()

include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-fopenmp" COMPILER_SUPPORT_OMP)
message("COMPILER_SUPPORT_OMP ${COMPILER_SUPPORT_OMP}")

if (WIN32)
    set(SIMD_FLAGS "/arch:AVX2" "/arch:FMA")
else ()
    set(SIMD_FLAGS "-march=native" "-mtune=native" "-fopenmp")
endif (WIN32)

install(FILES config.ini
        DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR})

add_subdirectory(libs)
add_subdirectory(examples)

function(print_cmake_variables_matching pattern)
    get_cmake_property(_variableNames VARIABLES)
    foreach (_variableName ${_variableNames})
        if (_variableName MATCHES "${pattern}")
            message(STATUS "${_variableName}=${${_variableName}}")
        endif ()
    endforeach ()
endfunction()

message(STATUS "")
message(STATUS "Generated with config types: ${CMAKE_CONFIGURATION_TYPES}")
message(STATUS "PROJECT_NAME: ${PROJECT_NAME}")
message(STATUS "PROJECT_SOURCE_DIR: ${PROJECT_SOURCE_DIR}")
message(STATUS "CMAKE_CURRENT_BINARY_DIR: ${CMAKE_CURRENT_BINARY_DIR}")
message(STATUS "CMAKE_SOURCE_DIR: ${CMAKE_SOURCE_DIR}")
message(STATUS "CMAKE_CXX_FLAGS_DEBUG: ${CMAKE_CXX_FLAGS_DEBUG}")
message(STATUS "CMAKE_CXX_FLAGS_RELEASE: ${CMAKE_CXX_FLAGS_RELEASE}")
message(STATUS "CMAKE_INSTALL_PREFIX: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "CMAKE_INSTALL_LIBDIR: ${CMAKE_INSTALL_LIBDIR}")
message(STATUS "CMAKE_INSTALL_BINDIR: ${CMAKE_INSTALL_BINDIR}")
message(STATUS "CMAKE_INSTALL_INCLUDEDIR: ${CMAKE_INSTALL_INCLUDEDIR}")

print_cmake_variables_matching("^TORCH")
print_cmake_variables_matching("^OpenCV")
print_cmake_variables_matching("^EIGEN3")
print_cmake_variables_matching("^xtensor")
print_cmake_variables_matching("^xsimd")
print_cmake_variables_matching("^xtl")
print_cmake_variables_matching("^OpenBLAS")
print_cmake_variables_matching("^BLAS")
print_cmake_variables_matching("^TBB")

message(STATUS "")
