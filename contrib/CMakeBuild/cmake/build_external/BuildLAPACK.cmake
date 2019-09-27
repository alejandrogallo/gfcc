#
# This file will build Netlib's LAPACK distribution over an existing BLAS
# installation. To do this we use a mock superbuild in case we need to build
# BLAS for the user.
#
find_or_build_dependency(BLAS)
package_dependency(BLAS DEPENDENCY_PATHS)
enable_language(C Fortran)

ExternalProject_Add(LAPACK_External
        SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/LAPACK
        CMAKE_ARGS ${DEPENDENCY_CMAKE_OPTIONS}
        BUILD_ALWAYS 1
        INSTALL_COMMAND $(MAKE) install DESTDIR=${STAGE_DIR}
        CMAKE_CACHE_ARGS ${CORE_CMAKE_LISTS}
                         ${CORE_CMAKE_STRINGS}
                         ${DEPENDENCY_PATHS}
        )
add_dependencies(LAPACK_External BLAS_External)
