cmake_minimum_required(VERSION ${CMAKE_VERSION})
project(GlobalArrays VERSION 5.7 LANGUAGES C Fortran)

include(DependencyMacros)
include(ExternalProject)
foreach(depend NWX_MPI CBLAS LAPACKE)#ScaLAPACK)
    find_dependency(${depend})
    makify_dependency(${depend} ${depend}_INCS ${depend}_LIBS)
endforeach()

#string(REPLACE ";" " " GA_ScaLAPACK_LIBRARIES "${ScaLAPACK_LIBRARIES}")
#set(GA_SCALAPACK "--with-scalapack8=${GA_ScaLAPACK_LIBRARIES}")

set(GA_BLAS "--with-blas8=${CBLAS_LIBS}")
if(BLAS_INT4)
    set(GA_BLAS "--with-blas4=${CBLAS_LIBS}")
endif()

set(GA_LAPACK "--with-lapack=${LAPACKE_LIBS}")

is_valid(SCALAPACK_LIBRARIES SCALAPACK_LIBS_SET)
if (SCALAPACK_LIBS_SET) 
    set(GA_SCALAPACK "--with-scalapack8=${SCALAPACK_LIBS}")
    if(BLAS_INT4)
        set(GA_SCALAPACK "--with-scalapack=${SCALAPACK_LIBS}")
    endif()    
else()
    set(GA_SCALAPACK "--without-scalapack")
endif()

set(GA_MPI "--with-mpi=${NWX_MPI_INCS} ${NWX_MPI_LIBS}")

set(GA_SYSVSHMEM "ARMCI_DEFAULT_SHMMAX_UBOUND=131072")

if (USE_OFFLOAD)
    set(GA_OFFLOAD "INTEL_64ALIGN=1")
endif()

set(GA_CXXFLAGS "${CMAKE_CXX_FLAGS} -fPIC")
set(GA_CFLAGS "${CMAKE_C_FLAGS} -fPIC")
set(GA_FFLAGS "${CMAKE_Fortran_Flags} -fPIC")

string (STRIP ${GA_CXXFLAGS} GA_CXXFLAGS)
string (STRIP ${GA_CFLAGS} GA_CFLAGS)
string (STRIP ${GA_FFLAGS} GA_FFLAGS)


# Build GA
set(GA_INSTALL_DIR ${STAGE_DIR}${CMAKE_INSTALL_PREFIX})
ExternalProject_Add(GlobalArrays_External
        URL https://github.com/GlobalArrays/ga/releases/download/v${PROJECT_VERSION}/ga-${PROJECT_VERSION}.tar.gz
        CONFIGURE_COMMAND                      CC=${CMAKE_C_COMPILER}
        CFLAGS=${GA_CFLAGS}
        CXX=${CMAKE_CXX_COMPILER}
        CXXFLAGS=${GA_CXXFLAGS}
        F77=${CMAKE_Fortran_COMPILER}
        FFLAGS=${GA_FFLAGS} ./configure --with-tcgmsg
                                      ${GA_MPI}
                                      --enable-underscoring
                                      --disable-mpi-tests
                                      --prefix=${GA_INSTALL_DIR}
                                      ${GA_SCALAPACK}
                                      ${GA_BLAS}
                                      ${GA_LAPACK}
                                      ${ARMCI_NETWORK}
                                      ${GA_OFFLOAD}
                                      ${GA_SYSVSHMEM}
                 
        BUILD_IN_SOURCE 1
        #LOG_CONFIGURE 1
        #LOG_BUILD 1
        )
