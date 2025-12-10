cmake_minimum_required(VERSION 3.24)

project(tatami_python
    VERSION 0.1.0
    DESCRIPTION "tatami bindings for abstract Python matrices"
    LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

add_library(tatami_python INTERFACE)
add_library(tatami::tatami_python ALIAS tatami_python)

option(TATAMI_PYTHON_FETCH_EXTERN "Automatically fetch tatami_python's external dependencies." ON)
if(TATAMI_PYTHON_FETCH_EXTERN)
    add_subdirectory(extern)
else()
    find_package(tatami_tatami 4.0.0 CONFIG REQUIRED)
    find_package(tatami_tatami_chunked 2.0.0 CONFIG REQUIRED)
    find_package(ltla_sanisizer 0.1.0 CONFIG REQUIRED)
endif()

target_link_libraries(tatami_python INTERFACE tatami::tatami tatami::tatami_chunked ltla::sanisizer)

# Switch between include directories depending on whether the downstream is
# using the build directly or is using the installed package.
include(GNUInstallDirs)
target_include_directories(tatami_python
    INTERFACE
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/tatami_tatami_python>"
)

# Installing for find_package.
include(CMakePackageConfigHelpers)

install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/tatami_tatami_python)

install(TARGETS tatami_python
    EXPORT tatami_pythonTargets)

install(EXPORT tatami_pythonTargets
    FILE tatami_tatami_pythonTargets.cmake
    NAMESPACE tatami::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/tatami_tatami_python)

configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/Config.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/tatami_tatami_pythonConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/tatami_tatami_python)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/tatami_tatami_pythonConfigVersion.cmake
    COMPATIBILITY SameMajorVersion)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/tatami_tatami_pythonConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/tatami_tatami_pythonConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/tatami_tatami_python)
