#
# This file will build Netlib's LAPACKE distribution over an existing CBLAS
# installation. To do this we use a mock superbuild in case we need to build
# CBLAS for the user.
#
find_or_build_dependency(BLAS)
find_or_build_dependency(LAPACK)
package_dependency(BLAS DEPENDENCY_PATHS)
package_dependency(LAPACK DEPENDENCY_PATHS)
enable_language(C Fortran)

ExternalProject_Add(LAPACKE_External
        SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/LAPACKE
        CMAKE_ARGS ${DEPENDENCY_CMAKE_OPTIONS}
                   -DSTAGE_DIR=${STAGE_DIR}
        BUILD_ALWAYS 1
        INSTALL_COMMAND $(MAKE) DESTDIR=${STAGE_DIR}
        CMAKE_CACHE_ARGS ${CORE_CMAKE_LISTS}
                         ${CORE_CMAKE_STRINGS}
                         ${DEPENDENCY_PATHS}
        )
add_dependencies(LAPACKE_External BLAS_External LAPACK_External)
