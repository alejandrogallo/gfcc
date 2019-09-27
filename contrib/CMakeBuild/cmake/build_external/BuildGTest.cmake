ExternalProject_Add(GTest_External
    URL https://github.com/google/googletest/archive/release-1.8.0.tar.gz
    CMAKE_ARGS ${DEPENDENCY_CMAKE_OPTIONS}
    INSTALL_COMMAND ${CMAKE_MAKE_PROGRAM} install DESTDIR=${STAGE_DIR}
)

