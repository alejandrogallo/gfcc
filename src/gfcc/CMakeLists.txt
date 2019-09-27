cmake_minimum_required(VERSION ${CMAKE_VERSION})
project(gfcc VERSION ${PROJECT_VERSION} LANGUAGES CXX)
include(TargetMacros) #For nwchemex_add_library

#Add the current directory's sources to the list
set(gfcc_SRCS gf_ccsd.cpp)

#Add the current directory's header files to the list
set(gfcc_INCLUDES
    ccsd_util.hpp       diis.hpp        hf_tamm_common.hpp
    cd_ccsd_common.hpp  gf_guess.hpp    input_parser.hpp
    cd_svd_ga.hpp       gf_ccsd_cs.hpp  hartree_fock_tamm.hpp  linalg.hpp
    gf_diis_cs.hpp  hf_common.hpp
   )

set(gfcc_DEFINITIONS )
set(gfcc_LFLAGS )

nwchemex_add_library(gfcc gfcc_SRCS gfcc_INCLUDES gfcc_DEFINTIONS gfcc_LFLAGS)


