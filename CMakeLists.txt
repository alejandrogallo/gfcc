
cmake_minimum_required(VERSION 3.13)
project(gfcc VERSION 1.0.0 LANGUAGES C CXX Fortran)
include(${PROJECT_SOURCE_DIR}/cmake/check_user_env.cmake)
set(NWX_PROJECTS gfcc)

set(gfcc_DEPENDENCIES NWX_MPI MSGSL Eigen3 GlobalArrays LibInt2 CBLAS LAPACKE HPTT BLIS tamm)

if(USE_CUDA) 
    list(APPEND gfcc_DEPENDENCIES TALSH)
endif()

set(BUILD_METHODS OFF)

set(gfcc_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/gfcc)
set(gfcc_TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR}/driver/)

if(NOT CMAKE_BUILD_TYPE)
    set (CMAKE_BUILD_TYPE Release)
endif()

find_package(CMakeBuild)
build_nwchemex_module(${CMAKE_CURRENT_LIST_DIR})

