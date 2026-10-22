set(headers
    "common.h"
    "cpu.h"
    "naive.h"
    "efficient.h"
    "efficient_hardware_opt.h"
    "efficient_hardware_recursive.h"
    "radix_sort.h"
    "thrust.h"
    )

set(sources
    "common.cu"
    "cpu.cu"
    "naive.cu"
    "efficient.cu"
    "efficient_hardware_opt.cu"
    "efficient_hardware_recursive.cu"
    "radix_sort.cu"
    "thrust.cu"
    )

list(SORT headers)
list(SORT sources)

source_group(Headers FILES ${headers})
source_group(Sources FILES ${sources})

add_library(stream_compaction ${sources} ${headers})
if(MSVC)
    target_compile_options(stream_compaction PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>")
endif()
if(CMAKE_VERSION VERSION_LESS "3.23.0")
    set_target_properties(stream_compaction} PROPERTIES CUDA_ARCHITECTURES OFF)
elseif(CMAKE_VERSION VERSION_LESS "3.24.0")
    set_target_properties(stream_compaction PROPERTIES CUDA_ARCHITECTURES all-major)
else()
    set_target_properties(stream_compaction PROPERTIES CUDA_ARCHITECTURES native)
endif()
set(device_debug_sources ${sources})
list(REMOVE_ITEM device_debug_sources "thrust.cu")
set_property(SOURCE ${device_debug_sources} APPEND PROPERTY COMPILE_OPTIONS
    "$<$<AND:$<CONFIG:Debug,RelWithDebInfo>,$<COMPILE_LANGUAGE:CUDA>>:-G;-src-in-ptx>")
# Unoptimized Thrust/CUB kernels can exhaust the default GPU stack with -G.
set_property(SOURCE thrust.cu APPEND PROPERTY COMPILE_OPTIONS
    "$<$<AND:$<CONFIG:Debug,RelWithDebInfo>,$<COMPILE_LANGUAGE:CUDA>>:-lineinfo;-src-in-ptx>")
target_compile_options(stream_compaction PRIVATE "$<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-lineinfo;-src-in-ptx>")
