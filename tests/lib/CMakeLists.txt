cmake_minimum_required(VERSION 3.24)

project(mattress
    VERSION 1.0.0
    DESCRIPTION "Building the mattress shared library"
    LANGUAGES CXX)

# Defining the targets.
find_package(pybind11 CONFIG)

# pybind11 method:
pybind11_add_module(mattress
    src/common.cpp
)

target_include_directories(mattress PRIVATE "${ASSORTHEAD_INCLUDE_DIR}")
target_include_directories(mattress PRIVATE "../../include")

set_property(TARGET mattress PROPERTY CXX_STANDARD 17)

target_link_libraries(mattress PRIVATE pybind11::pybind11)

set_target_properties(mattress PROPERTIES
    OUTPUT_NAME lib_mattress
    PREFIX ""
)
