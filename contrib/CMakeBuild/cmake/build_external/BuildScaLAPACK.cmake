#
# This file will build Netlib's ScaLAPACK distribution over an existing CBLAS
# and LAPACKE installation. To do this we use a mock superbuild in case we need
# to build CBLAS or LAPACKE for the user.
#
find_or_build_dependency(CBLAS _was_Found)
find_or_build_dependency(LAPACKE _was_Found)
enable_language(C Fortran)

ExternalProject_Add(ScaLAPACK_External
        SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/ScaLAPACK
        CMAKE_ARGS ${DEPENDENCY_CMAKE_OPTIONS}
                   -DSTAGE_DIR=${STAGE_DIR}
        BUILD_ALWAYS 1
        INSTALL_COMMAND $(MAKE) DESTDIR=${STAGE_DIR}
        CMAKE_CACHE_ARGS ${CORE_CMAKE_LISTS}
                         ${CORE_CMAKE_STRINGS}
        )
add_dependencies(ScaLAPACK_External LAPACKE_External CBLAS_External)


