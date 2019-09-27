cmake_minimum_required(VERSION ${CMAKE_VERSION})
project(gfcc VERSION ${PROJECT_VERSION} LANGUAGES CXX)
include(TargetMacros) #For nwchemex_add_library

#Add the current directory's sources to the list
set(gfcc_SRCS gf_ccsd.cpp)

#Add the current directory's header files to the list
set(gfcc_INCLUDES
    gf_ccsd.cpp  gf_ccsd_cs.hpp  gf_diis_cs.hpp  gf_guess.hpp
    contrib/ccsd_util.hpp          contrib/cd_svd_ga.hpp  
    contrib/cd_ccsd_common.hpp     contrib/diis.hpp       
    contrib/hf_common.hpp          
    contrib/hartree_fock_tamm.hpp  contrib/hf_tamm_common.hpp  
    contrib/input_parser.hpp       contrib/linalg.hpp
   )

set(gfcc_DEFINITIONS )
set(gfcc_LFLAGS )

nwchemex_add_library(gfcc gfcc_SRCS gfcc_INCLUDES gfcc_DEFINTIONS gfcc_LFLAGS)


