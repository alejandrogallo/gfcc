cmake_minimum_required(VERSION ${CMAKE_VERSION})
project(gfcc-Test VERSION 0.0.0 LANGUAGES CXX)
find_package(gfcc REQUIRED)
include(TargetMacros)

include_directories(${CMAKE_SOURCE_DIR}/..)
add_mpi_unit_test(GF_CCSD 2 "${CMAKE_SOURCE_DIR}/../inputs/co.nwx")



