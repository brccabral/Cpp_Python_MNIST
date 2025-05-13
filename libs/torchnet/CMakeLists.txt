find_package(Torch REQUIRED)

add_library(torchnet INTERFACE)
target_sources(torchnet
        INTERFACE
        FILE_SET HEADERS
        FILES include/TorchNet/torchnet.hpp
        BASE_DIRS include
)
target_link_libraries(torchnet INTERFACE "${TORCH_LIBRARIES}")

if (USE_SIMD)
    target_compile_options(torchnet INTERFACE ${SIMD_FLAGS})
    target_link_options(torchnet INTERFACE ${SIMD_FLAGS})
endif ()
