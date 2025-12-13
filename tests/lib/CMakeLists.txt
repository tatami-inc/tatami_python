cmake_minimum_required(VERSION 3.24)

project(tatami_python_test
    VERSION 1.0.0
    DESCRIPTION "Building the tatami_python_test shared library"
    LANGUAGES CXX)

# Defining the targets.
find_package(pybind11 CONFIG)

# pybind11 method:
pybind11_add_module(tatami_python_test
    src/bindings.cpp
)

target_include_directories(tatami_python_test PRIVATE "../../build/_deps/sanisizer-src/include")
target_include_directories(tatami_python_test PRIVATE "../../build/_deps/subpar-src/include")
target_include_directories(tatami_python_test PRIVATE "../../build/_deps/tatami-src/include")
target_include_directories(tatami_python_test PRIVATE "../../build/_deps/tatami_chunked-src/include")
target_include_directories(tatami_python_test PRIVATE "../../include")

target_compile_definitions(tatami_python_test PRIVATE TEST_CUSTOM_PARALLEL=1)
target_compile_options(tatami_python_test PRIVATE -Wall -Wpedantic -Wextra -Werror)

set_property(TARGET tatami_python_test PROPERTY CXX_STANDARD 17)

target_link_libraries(tatami_python_test PRIVATE pybind11::pybind11)

set_target_properties(tatami_python_test PROPERTIES
    OUTPUT_NAME lib_tatami_python_test
    PREFIX ""
)
