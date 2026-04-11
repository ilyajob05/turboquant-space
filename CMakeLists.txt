cmake_minimum_required(VERSION 3.18...4.31)

project(turboquant_space LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

list(APPEND CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/.venv")

if(WIN32)
    set(PYTHON_PATH "${CMAKE_CURRENT_SOURCE_DIR}/.venv/Scripts/python.exe")
else()
    set(PYTHON_PATH "${CMAKE_CURRENT_SOURCE_DIR}/.venv/bin/python")
endif()

execute_process(
    COMMAND "${PYTHON_PATH}" -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE PYBIND11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_VARIABLE PY_ERROR
)

if(NOT PYBIND11_DIR)
    message(FATAL_ERROR "pybind11 not found in .venv! Error: ${PY_ERROR}")
endif()

message(${CMAKE_PREFIX_PATH})
message(${PYBIND11_DIR})

# find_package(pybind11 REQUIRED)
find_package(pybind11 REQUIRED PATHS ${PYBIND11_DIR} NO_DEFAULT_PATH)

pybind11_add_module(_turboquant python/turboquant/bindings.cpp)

target_include_directories(_turboquant PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)

# --- OpenMP (optional) ------------------------------------------------------
# Apple Clang ships without OpenMP; find_package can't locate Homebrew libomp
# by itself, so we provide hints when running on macOS.
if(APPLE)
    execute_process(
        COMMAND brew --prefix libomp
        OUTPUT_VARIABLE LIBOMP_PREFIX
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
        RESULT_VARIABLE LIBOMP_BREW_RC
    )
    if(LIBOMP_BREW_RC EQUAL 0 AND EXISTS "${LIBOMP_PREFIX}")
        set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I${LIBOMP_PREFIX}/include")
        set(OpenMP_CXX_LIB_NAMES "omp")
        set(OpenMP_omp_LIBRARY "${LIBOMP_PREFIX}/lib/libomp.dylib")
    endif()
endif()

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found: parallel batch ops enabled")
    target_link_libraries(_turboquant PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(_turboquant PRIVATE TURBOQUANT_HAVE_OPENMP)
else()
    message(STATUS "OpenMP not found: batch ops will run single-threaded")
endif()

set_target_properties(_turboquant PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/python/turboquant"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/python/turboquant"
)

if(MSVC)
    target_compile_options(_turboquant PRIVATE /O2 /fp:fast /arch:AVX2)
else()
    target_compile_options(_turboquant PRIVATE
        -O3
        -ffast-math
        -fno-finite-math-only
    )
endif()

target_compile_definitions(_turboquant PRIVATE
    $<$<CONFIG:Debug>:DEBUG_MODE>
    $<$<CONFIG:Release>:NDEBUG>
)

# Install the compiled module next to the Python package sources so
# `from turboquant import _turboquant` resolves inside the installed wheel.
install(TARGETS _turboquant LIBRARY DESTINATION turboquant)
