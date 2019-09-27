#
# This file will build LibInt using a mock super-build incase Eigen3 needs to be
# built as well
#
find_or_build_dependency(Eigen3)
package_dependency(Eigen3 DEPENDENCY_PATHS)
set(TEST_LIBINT FALSE)
if(${PROJECT_NAME} STREQUAL "TestBuildLibInt")
    set(TEST_LIBINT TRUE)
endif()

set(LIBINT_VERSION 2.6.0)
set(LIBINT_URL https://github.com/evaleev/libint)
set(LIBINT_TAR ${LIBINT_URL}/releases/download/v${LIBINT_VERSION})
set(LIBINT_TAR ${LIBINT_TAR}/libint-${LIBINT_VERSION})
if(TEST_LIBINT)
    #Grab the small version of libint for testing purposes
    set(LIBINT_TAR ${LIBINT_TAR}-test-mpqc4.tgz)
else()
    set(LIBINT_TAR ${LIBINT_TAR}.tgz)
endif()

# append platform-specific optimization options for non-Debug builds
set(LIBINT_EXTRA_FLAGS "-Wno-unused-variable")
if(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
    set(LIBINT_EXTRA_FLAGS "-xHost ${LIBINT_EXTRA_FLAGS}")
elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "ppc64le")
    set(LIBINT_EXTRA_FLAGS "-mtune=native ${LIBINT_EXTRA_FLAGS}")
else()
    set(LIBINT_EXTRA_FLAGS "-march=native ${LIBINT_EXTRA_FLAGS}")
endif()
set(CXX_FLAGS_INIT "${CMAKE_CXX_FLAGS_INIT} ${LIBINT_EXTRA_FLAGS}")

ExternalProject_Add(LibInt2_External
        URL ${LIBINT_TAR}
        CMAKE_ARGS ${DEPENDENCY_CMAKE_OPTIONS} -DCMAKE_CXX_FLAGS_INIT=${CXX_FLAGS_INIT}
        INSTALL_COMMAND ${CMAKE_MAKE_PROGRAM} install DESTDIR=${STAGE_DIR}
        CMAKE_CACHE_ARGS ${CORE_CMAKE_LISTS}
        ${CORE_CMAKE_STRINGS}
        )

add_dependencies(LibInt2_External Eigen3_External)

