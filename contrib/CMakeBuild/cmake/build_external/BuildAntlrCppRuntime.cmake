set(ANTLR_PATCH_FILE
        ${PROJECT_SOURCE_DIR}/CMakeBuild/cmake/external/patches/antlr_cmakelists.patch)

ExternalProject_Add(AntlrCppRuntime${TARGET_SUFFIX}
        URL http://www.antlr.org/download/antlr4-cpp-runtime-4.7.1-source.zip
        PATCH_COMMAND patch < ${ANTLR_PATCH_FILE}
        CMAKE_ARGS ${DEPENDENCY_CMAKE_OPTIONS}
                   -DWITH_DEMO=OFF -DWITH_LIBCXX=OFF
        INSTALL_COMMAND ${CMAKE_MAKE_PROGRAM} install DESTDIR=${STAGE_DIR}
)

