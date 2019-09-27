cmake_minimum_required(VERSION ${CMAKE_VERSION})
project(gfcc-Test VERSION 0.0.0 LANGUAGES CXX)
find_package(gfcc REQUIRED)
include(TargetMacros)
add_mpi_unit_test(GF_CCSD_CS 2 "${CMAKE_SOURCE_DIR}/../inputs/co.nwx")



