cmake_minimum_required(VERSION 3.6)
project(TestBuildCBLAS VERSION 1.0.0 LANGUAGES CXX)
find_package(CMakeBuild)
set(TestBuildCBLAS_DEPENDENCIES CBLAS)
set(BUILD_BLAS TRUE)
set(BUILD_CBLAS TRUE)
build_nwchemex_module(${CMAKE_CURRENT_LIST_DIR})
