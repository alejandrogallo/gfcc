#
# This file actually builds BLIS
#
cmake_minimum_required(VERSION ${CMAKE_VERSION})
project(BLIS LANGUAGES C CXX)
include(DependencyMacros)
include(ExternalProject)
include(UtilityMacros)

string_concat(CMAKE_CXX_FLAGS_RELEASE "" " " BLIS_FLAGS)

if(CMAKE_POSITION_INDEPENDENT_CODE)
    set(FPIC_LIST "-fPIC")
    string_concat(FPIC_LIST "" " " BLIS_FLAGS)
endif()

set(BLIS_TAR https://github.com/flame/blis/archive/0.7.0.tar.gz)

set(BLIS_OPT_FLAGS -march=native)
if(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
    set(BLIS_OPT_FLAGS -xHost)
elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "ppc64le")
    set(BLIS_OPT_FLAGS -mtune=native)
endif()

set(BLIS_OPT_FLAGS "${BLIS_FLAGS} ${BLIS_OPT_FLAGS}")

set(BLIS_INT_FLAGS -i 64 -b 64 -t openmp)# --enable-cblas

if(BLAS_INT4)
    set(BLIS_INT_FLAGS -i 32 -b 32 -t openmp)# --enable-cblas
endif()

ExternalProject_Add(BLIS_External
        URL ${BLIS_TAR}
        CONFIGURE_COMMAND ./configure --prefix=${CMAKE_INSTALL_PREFIX}
                                      CXX=${CMAKE_CXX_COMPILER}
                                      CC=${CMAKE_C_COMPILER}
                                      CFLAGS=${BLIS_OPT_FLAGS}
                                      ${BLIS_INT_FLAGS}
                                      ${BLIS_CONFIG_HW}
        INSTALL_COMMAND ${CMAKE_MAKE_PROGRAM} install DESTDIR=${STAGE_DIR}
        BUILD_IN_SOURCE 1
)
