option(CH_ENABLE_MODULE_CUDSS "Enable the Chrono cuDSS module" OFF)

if(NOT CH_ENABLE_MODULE_CUDSS)
    return()
endif()

message(STATUS "\n==== Chrono cuDSS module ====\n")

enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)

find_path(CUDSS_INCLUDE_DIR
    NAMES cudss.h
    HINTS
        ${CUDAToolkit_INCLUDE_DIRS}
        /usr/local/cuda/include
        /usr/include
)

find_library(CUDSS_LIBRARY
    NAMES cudss
    HINTS
        ${CUDAToolkit_LIBRARY_DIR}
        /usr/lib/x86_64-linux-gnu
        /usr/local/cuda/lib64
)

if(NOT CUDSS_INCLUDE_DIR OR NOT CUDSS_LIBRARY)
    message(FATAL_ERROR "cuDSS not found. Set CUDSS_INCLUDE_DIR and CUDSS_LIBRARY manually.")
endif()

message(STATUS "cuDSS include: ${CUDSS_INCLUDE_DIR}")
message(STATUS "cuDSS library: ${CUDSS_LIBRARY}")

# Create an imported target so the solver library links cleanly
if(NOT TARGET cuDSS::cuDSS)
    add_library(cuDSS::cuDSS SHARED IMPORTED)
    set_target_properties(cuDSS::cuDSS PROPERTIES
        IMPORTED_LOCATION             "${CUDSS_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${CUDSS_INCLUDE_DIR}"
    )
endif()

set(Chrono_CuDSS_HEADERS
    ChApiCuDSS.h
    ChSolverCuDSS.h
)

set(Chrono_CuDSS_SOURCES
    ChSolverCuDSS.cpp
)

source_group("" FILES ${Chrono_CuDSS_HEADERS} ${Chrono_CuDSS_SOURCES})

# Add the Chrono_cudss library
add_library(Chrono_cudss
    ${Chrono_CuDSS_SOURCES}
    ${Chrono_CuDSS_HEADERS}
)
add_library(Chrono::cudss ALIAS Chrono_cudss)

set_target_properties(Chrono_cudss PROPERTIES DEBUG_POSTFIX ${CH_DEBUG_POSTFIX})

if(CH_WHOLE_PROG_OPT)
    set_target_properties(Chrono_cudss PROPERTIES COMPILE_FLAGS "/GL")
    set_target_properties(Chrono_cudss PROPERTIES LINK_FLAGS    "/LTCG")
endif()

if(CH_STATIC)
    set_target_properties(Chrono_cudss PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

if(MSVC)
    set_target_properties(Chrono_cudss PROPERTIES MSVC_RUNTIME_LIBRARY ${CH_MSVC_RUNTIME_LIBRARY})
endif()

target_link_libraries(Chrono_cudss
    Chrono_core
    cuDSS::cuDSS
    CUDA::cudart
)

target_compile_definitions(Chrono_cudss PRIVATE "CH_API_COMPILE_CUDSS")

# Install
install(TARGETS Chrono_cudss
    EXPORT   ChronoTargets
    RUNTIME  DESTINATION bin
    LIBRARY  DESTINATION lib
    ARCHIVE  DESTINATION lib
    INCLUDES DESTINATION include/chrono_cudss
)

install(FILES ${Chrono_CuDSS_HEADERS}
    DESTINATION include/chrono_cudss
)