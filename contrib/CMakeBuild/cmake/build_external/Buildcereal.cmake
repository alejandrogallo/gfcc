ExternalProject_Add(cereal_External
    GIT_REPOSITORY https://github.com/USCiLab/cereal
    GIT_TAG 51cbda5f30e56c801c07fe3d3aba5d7fb9e6cca4
    CMAKE_ARGS ${DEPENDENCY_CMAKE_OPTIONS}
               -DSKIP_PORTABILITY_TEST=TRUE
               -DJUST_INSTALL_CEREAL=TRUE
    INSTALL_COMMAND ${CMAKE_MAKE_PROGRAM} install DESTDIR=${STAGE_DIR}
)

