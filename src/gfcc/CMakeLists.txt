cmake_minimum_required(VERSION ${CMAKE_VERSION})
project(gfcc VERSION ${PROJECT_VERSION} LANGUAGES CXX)
include(TargetMacros) #For nwchemex_add_library

#Add the current directory's sources to the list
set(gfcc_SRCS gf_ccsd.cpp)

#Add the current directory's header files to the list
set(gfcc_INCLUDES
    gf_ccsd.hpp  gf_diis.hpp  gf_guess.hpp gfccsd_ip.hpp
    contrib/ccsd_util.hpp          contrib/cd_svd_ga.hpp  
    contrib/cd_ccsd_common.hpp     contrib/diis.hpp       
    contrib/hf_common.hpp          contrib/scf_guess.hpp
    contrib/hartree_fock_tamm.hpp  contrib/hf_tamm_common.hpp  
    contrib/input_parser.hpp       contrib/linalg.hpp contrib/molden.hpp
   )

set(gfcc_DEFINITIONS )
set(gfcc_LFLAGS )

nwchemex_add_library(gfcc gfcc_SRCS gfcc_INCLUDES gfcc_DEFINTIONS gfcc_LFLAGS)
target_include_directories(gfcc PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../..)


