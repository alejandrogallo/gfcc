#
# This file will build BLIS.
#

# find_or_build_dependency(CBLAS)

is_valid_and_true(BLIS_CONFIG __set)
if (NOT __set)
    message(STATUS "BLIS_CONFIG not set, will auto-detect")
    set(BLIS_CONFIG_HW "auto")
else()
    message(STATUS "BLIS_CONFIG set to ${BLIS_CONFIG}")
    set(BLIS_CONFIG_HW ${BLIS_CONFIG})
endif()

ExternalProject_Add(BLIS_External
        SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/BLIS
        CMAKE_ARGS ${DEPENDENCY_CMAKE_OPTIONS}
                   -DBLIS_CONFIG_HW=${BLIS_CONFIG_HW}
                   -DSTAGE_DIR=${STAGE_DIR}
                   -DTEST_BLIS=${TEST_BLIS}
        BUILD_ALWAYS 1
        INSTALL_COMMAND $(MAKE) DESTDIR=${STAGE_DIR}
        CMAKE_CACHE_ARGS ${CORE_CMAKE_LISTS}
                         ${CORE_CMAKE_STRINGS}
                         ${DEPENDENCY_PATHS}
        )


# add_dependencies(BLIS_External CBLAS_External)
                 