#
# This file will build Netlib's ScaLAPACK distribution over an existing CBLAS
# and LAPACKE installation. To do this we use a mock superbuild in case we need
# to build CBLAS or LAPACKE for the user.
#
enable_language(C Fortran)
foreach(depend CBLAS LAPACKE)
    find_or_build_dependency(${depend})
    package_dependency(${depend} DEPENDENCY_PATHS)
endforeach()

ExternalProject_Add(ScaLAPACK_External
        SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/ScaLAPACK
        CMAKE_ARGS ${DEPENDENCY_CMAKE_OPTIONS}
                   -DSTAGE_DIR=${STAGE_DIR}
        #BUILD_ALWAYS 1
        INSTALL_COMMAND $(MAKE) DESTDIR=${STAGE_DIR}
        CMAKE_CACHE_ARGS ${CORE_CMAKE_LISTS}
                         ${CORE_CMAKE_STRINGS}
                         ${DEPENDENCY_PATHS}
        )
add_dependencies(ScaLAPACK_External LAPACKE_External CBLAS_External)


