ExternalProject_Add(Eigen3_External
    URL https://gitlab.com/libeigen/eigen/-/archive/3.3.7/eigen-3.3.7.tar.gz
    CMAKE_ARGS ${DEPENDENCY_CMAKE_OPTIONS} 
        INSTALL_COMMAND ${CMAKE_MAKE_PROGRAM} install DESTDIR=${STAGE_DIR}
    )

