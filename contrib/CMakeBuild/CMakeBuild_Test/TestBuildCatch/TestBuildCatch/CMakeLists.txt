cmake_minimum_required(VERSION ${CMAKE_VERSION})
project(TestBuildCatch VERSION ${PROJECT_VERSION} LANGUAGES CXX)
include(TargetMacros)
set(TestBuildCatchSRCS TestBuildCatch.cpp)
set(TestBuildCatchHEADERS TestBuildCatch.hpp)
nwchemex_add_library(TestBuildCatch TestBuildCatchSRCS TestBuildCatchHEADERS "" "")
