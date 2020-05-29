cmake_minimum_required(VERSION 3.13)
project(CMakeBuild VERSION 0.0.0 LANGUAGES C CXX Fortran)
include(${CMAKE_CURRENT_LIST_DIR}/cmake/macros/BuildNWChemExModule.cmake)

if(NOT DEFINED BUILD_TESTS)
    set(BUILD_TESTS OFF)
endif()
set(BUILD_METHODS OFF)

set(CMakeBuild_DEPENDENCIES NWX_Catch)
build_nwchemex_module(${CMAKE_CURRENT_LIST_DIR})
