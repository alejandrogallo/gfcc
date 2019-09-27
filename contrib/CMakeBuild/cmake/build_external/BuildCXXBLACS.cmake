
find_or_build_dependency(CBLAS _was_Found)
find_or_build_dependency(LAPACKE _was_Found)
enable_language(C Fortran)

ExternalProject_Add(CXXBLACS_External
    GIT_REPOSITORY https://github.com/wavefunction91/CXXBLACS.git
    UPDATE_DISCONNECTED 1
    CMAKE_ARGS ${DEPENDENCY_CMAKE_OPTIONS} -Dlinalg_LIBRARIES="${LAPACKE_LIBRARIES}" -Dscalapack_LIBRARIES="${SCALAPACK_LIBRARIES}"
    INSTALL_COMMAND ${CMAKE_MAKE_PROGRAM} install DESTDIR=${STAGE_DIR}
    CMAKE_CACHE_ARGS ${CORE_CMAKE_LISTS}
                     ${CORE_CMAKE_STRINGS}
)

