cmake_minimum_required(VERSION 3.18...4.31)

project(turboquant_space LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE PYBIND11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_VARIABLE PY_ERROR
    RESULT_VARIABLE PY_RESULT
)

if(NOT PY_RESULT EQUAL 0 OR NOT PYBIND11_DIR)
    message(FATAL_ERROR "pybind11 not found via ${Python_EXECUTABLE}! Error: ${PY_ERROR}")
endif()

find_package(pybind11 REQUIRED PATHS ${PYBIND11_DIR} NO_DEFAULT_PATH)

pybind11_add_module(_turboquant python/turboquant/bindings.cpp)

target_include_directories(_turboquant PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)

# --- OpenMP (optional) ------------------------------------------------------
# Apple Clang ships without OpenMP. To avoid the "multiple libomp loaded"
# abort when torch is also imported, prefer torch's bundled libomp if
# available; otherwise fall back to Homebrew's libomp. We still need an
# OpenMP-capable header path for compilation.
if(APPLE)
    set(LIBOMP_INCLUDE_DIR "")
    set(LIBOMP_LIBRARY "")

    execute_process(
        COMMAND "${Python_EXECUTABLE}" -c "import os, torch; print(os.path.join(os.path.dirname(torch.__file__), 'lib', 'libomp.dylib'))"
        OUTPUT_VARIABLE TORCH_LIBOMP
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
        RESULT_VARIABLE TORCH_LIBOMP_RC
    )

    execute_process(
        COMMAND brew --prefix libomp
        OUTPUT_VARIABLE LIBOMP_BREW_PREFIX
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
        RESULT_VARIABLE LIBOMP_BREW_RC
    )
    if(LIBOMP_BREW_RC EQUAL 0 AND EXISTS "${LIBOMP_BREW_PREFIX}/include/omp.h")
        set(LIBOMP_INCLUDE_DIR "${LIBOMP_BREW_PREFIX}/include")
    endif()

    if(TORCH_LIBOMP_RC EQUAL 0 AND EXISTS "${TORCH_LIBOMP}")
        set(LIBOMP_LIBRARY "${TORCH_LIBOMP}")
        message(STATUS "Using torch-bundled libomp: ${LIBOMP_LIBRARY}")
    elseif(LIBOMP_BREW_RC EQUAL 0 AND EXISTS "${LIBOMP_BREW_PREFIX}/lib/libomp.dylib")
        set(LIBOMP_LIBRARY "${LIBOMP_BREW_PREFIX}/lib/libomp.dylib")
        message(STATUS "Using Homebrew libomp: ${LIBOMP_LIBRARY}")
    endif()

    if(LIBOMP_INCLUDE_DIR AND LIBOMP_LIBRARY)
        set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I${LIBOMP_INCLUDE_DIR}")
        set(OpenMP_CXX_LIB_NAMES "omp")
        set(OpenMP_omp_LIBRARY "${LIBOMP_LIBRARY}")
    endif()
endif()

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found: parallel batch ops enabled")
    target_link_libraries(_turboquant PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(_turboquant PRIVATE TURBOQUANT_HAVE_OPENMP)

    # On macOS, rewrite the libomp install name in the .so so dyld resolves it
    # via @rpath, and add rpaths to both torch's lib dir and Homebrew's libomp
    # dir. This way torch and turboquant share the same loaded libomp instance,
    # avoiding the "OMP: Error #15: ... already initialized" abort.
    if(APPLE AND LIBOMP_LIBRARY)
        # Search torch's bundled libomp first (relative to the .so's install
        # location: site-packages/turboquant/_turboquant.so → ../torch/lib).
        # If torch isn't installed, fall back to Homebrew's libomp via @loader_path.
        # @loader_path lets the .so find a sibling libomp.dylib if we ever ship one.
        set_target_properties(_turboquant PROPERTIES
            BUILD_WITH_INSTALL_RPATH TRUE
            INSTALL_RPATH "@loader_path/../torch/lib;@loader_path;/opt/homebrew/opt/libomp/lib;/usr/local/opt/libomp/lib"
        )
        add_custom_command(TARGET _turboquant POST_BUILD
            COMMAND install_name_tool -change
                "${LIBOMP_LIBRARY}" "@rpath/libomp.dylib"
                "$<TARGET_FILE:_turboquant>"
            COMMAND install_name_tool -change
                "/opt/llvm-openmp/lib/libomp.dylib" "@rpath/libomp.dylib"
                "$<TARGET_FILE:_turboquant>" || true
            COMMAND install_name_tool -change
                "/opt/homebrew/opt/libomp/lib/libomp.dylib" "@rpath/libomp.dylib"
                "$<TARGET_FILE:_turboquant>" || true
            COMMENT "Rewriting libomp install name to @rpath/libomp.dylib"
        )
    endif()
else()
    message(STATUS "OpenMP not found: batch ops will run single-threaded")
endif()

set_target_properties(_turboquant PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/python/turboquant"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/python/turboquant"
)

# TURBOQUANT_PORTABLE=ON  → conservative baseline for prebuilt wheels
#                           (x86-64-v3 on x64, armv8-a on arm64, /arch:AVX2 on MSVC).
# TURBOQUANT_PORTABLE=OFF → -march=native, tuned for the build machine.
# Default is OFF so local `pip install` / `uv sync` get max performance;
# cibuildwheel sets it to ON to produce portable wheels.
option(TURBOQUANT_PORTABLE "Build with a portable CPU baseline instead of -march=native" OFF)

if(MSVC)
    target_compile_options(_turboquant PRIVATE /O2 /fp:fast /arch:AVX2)
else()
    target_compile_options(_turboquant PRIVATE
        -O3
        -ffast-math
        -fno-finite-math-only
    )
    if(TURBOQUANT_PORTABLE)
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$")
            target_compile_options(_turboquant PRIVATE -march=x86-64-v3)
        elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
            target_compile_options(_turboquant PRIVATE -march=armv8-a)
        endif()
    else()
        target_compile_options(_turboquant PRIVATE -march=native)
    endif()
endif()

target_compile_definitions(_turboquant PRIVATE
    $<$<CONFIG:Debug>:DEBUG_MODE>
    $<$<CONFIG:Release>:NDEBUG>
)

# Install the compiled module next to the Python package sources so
# `from turboquant import _turboquant` resolves inside the installed wheel.
install(TARGETS _turboquant LIBRARY DESTINATION turboquant)
