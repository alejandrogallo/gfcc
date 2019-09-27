set(NWXBASE_MACROS ${CMAKE_CURRENT_LIST_DIR})

function(build_nwchemex_module SUPER_PROJECT_ROOT)

    #Set the environment up and pull-in macros we'll need
    include(${NWXBASE_MACROS}/SetPaths.cmake)
    set_paths() #Puts macro paths in module path
    include(OptionMacros)
    include(DependencyMacros)
    include(ExternalProject)
    include(UtilityMacros)

    #We require C++14 get it out of the way early
    option_w_default(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    option_w_default(BLAS_INT4 ON)
    option_w_default(CMAKE_CXX_EXTENSIONS OFF)
    option_w_default(CMAKE_BUILD_TYPE Release)

    if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        get_filename_component(__NWX_GCC_INSTALL_PREFIX "${CMAKE_Fortran_COMPILER}/../.." ABSOLUTE)
        set(NWX_GCC_TOOLCHAIN_FLAG "--gcc-toolchain=${__NWX_GCC_INSTALL_PREFIX}")
        message(STATUS "NWX_GCC_TOOLCHAIN_FLAG: ${NWX_GCC_TOOLCHAIN_FLAG}")
    endif()

    if(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
        set(NWX_EXTRA_FLAGS "-xHost")
    elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "ppc64le")
        set(NWX_EXTRA_FLAGS "-mtune=native")
        if(NWX_CUDA)
            #nvcc does not recgonize -mtune=power9
            set(NWX_EXTRA_FLAGS "-mtune=powerpc64le")
        endif()
    else()
        set(NWX_EXTRA_FLAGS "-march=native")
    endif()

    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall ${NWX_GCC_TOOLCHAIN_FLAG} ${NWX_EXTRA_FLAGS}")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -Wall ${NWX_GCC_TOOLCHAIN_FLAG} ${NWX_EXTRA_FLAGS}")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -O3 -Wall ${NWX_GCC_TOOLCHAIN_FLAG} ${NWX_EXTRA_FLAGS}")

    string(TOUPPER ${CMAKE_BUILD_TYPE} NWX_CMAKE_BUILD_TYPE)
    set(NWX_CXX_FLAGS CMAKE_CXX_FLAGS_${NWX_CMAKE_BUILD_TYPE})

    print_banner("Configuration Options")
    
    option_w_default(BUILD_SHARED_LIBS OFF)
    option_w_default(CMAKE_POSITION_INDEPENDENT_CODE TRUE)
    option_w_default(BUILD_TESTS ON)    #Should we build the tests?
    option_w_default(NWX_DEBUG_CMAKE TRUE) #Enable lots of extra CMake printing?
    option_w_default(CMAKE_EXPORT_COMPILE_COMMANDS ON)
    option_w_default(CMAKE_VERBOSE_MAKEFILE ${NWX_DEBUG_CMAKE})
    option_w_default(CMAKE_FIND_PACKAGE_NO_PACKAGE_REGISTRY TRUE)

    print_banner("NWChemEx Module Paths")

    option_w_default(NWX_PROJECTS ${PROJECT_NAME}) # List of modules to build
    foreach(__project ${NWX_PROJECTS})
        #Directory where the sub-project's source is located
        option_w_default(${__project}_SRC_DIR ${SUPER_PROJECT_ROOT}/${__project})
        #Includes should be relative to NWX_SRC_DIR without last directory
        get_filename_component(${__project}_INCLUDE_DIR "${${__project}_SRC_DIR}"
                DIRECTORY)
        #Directory where your tests are
        option_w_default(${__project}_TEST_DIR
                ${SUPER_PROJECT_ROOT}/${__project}_Test)
        #Name of variable containing your project's dependencies
        option_w_default(${__project}_DEPENDENCIES "")
    endforeach()

    set(NWX_CORE_OPTIONS CMAKE_CXX_COMPILER CMAKE_C_COMPILER
        CMAKE_Fortran_COMPILER CMAKE_BUILD_TYPE BUILD_SHARED_LIBS ${NWX_CXX_FLAGS}
        CMAKE_INSTALL_PREFIX CMAKE_CXX_STANDARD CMAKE_VERSION PROJECT_VERSION
        CMAKE_POSITION_INDEPENDENT_CODE CMAKE_VERBOSE_MAKEFILE CMAKE_CXX_EXTENSIONS
        CMAKE_FIND_PACKAGE_NO_PACKAGE_REGISTRY CMAKE_EXPORT_COMPILE_COMMANDS)

    #Make a list of all CMake variables that should be passed to all dependencies
    bundle_cmake_args(CORE_CMAKE_OPTIONS ${NWX_CORE_OPTIONS})

    bundle_cmake_list(CORE_CMAKE_LISTS CMAKE_PREFIX_PATH CMAKE_INSTALL_RPATH CMAKE_MODULE_PATH)

    bundle_cmake_strings(CORE_CMAKE_STRINGS ${NWX_CXX_FLAGS})

    bundle_cmake_args(DEPENDENCY_CMAKE_OPTIONS ${NWX_CORE_OPTIONS})

    if (SCALAPACK) 
        bundle_cmake_args(DEPENDENCY_CMAKE_OPTIONS SCALAPACK)
        set(TAMM_CXX_FLAGS "${TAMM_CXX_FLAGS} -DSCALAPACK")
    endif()

    string(FIND "${LAPACKE_LIBRARIES}" "mkl" FINDLAPACKE_mkl_found)
    string(FIND "${LAPACKE_LIBRARIES}" "essl" FINDLAPACKE_essl_found)

    if(NOT "${FINDLAPACKE_mkl_found}" STREQUAL "-1")
        string(FIND "${LAPACKE_LIBRARIES}" "ilp64" _mkl_ilp64_found)
        if(NOT "${_mkl_ilp64_found}" STREQUAL "-1")
            message(STATUS "DETECTED INTEL MKL ILP64 LIBS")
            set(BLAS_INT4 OFF CACHE BOOL "BLAS INT SIZE" FORCE)
            set(TAMM_CXX_FLAGS "${TAMM_CXX_FLAGS} -m64 -DMKL_ILP64" CACHE STRING "TAMM_CXX_FLAGS" FORCE)
        endif()              
    elseif(NOT "${FINDLAPACKE_essl_found}" STREQUAL "-1")
        string(FIND "${LAPACKE_LIBRARIES}" "essl6464" _essl_ilp64_found)
        string(FIND "${LAPACKE_LIBRARIES}" "esslsmp6464" _esslsmp_ilp64_found)
        if(NOT "${_essl_ilp64_found}" STREQUAL "-1" OR NOT "${_esslsmp_ilp64_found}" STREQUAL "-1")
            message(STATUS "DETECTED IBM ESSL ILP64 LIBS")
            set(BLAS_INT4 OFF CACHE BOOL "BLAS INT SIZE" FORCE)
            set(TAMM_CXX_FLAGS "${TAMM_CXX_FLAGS} -m64 -DLAPACK_ILP64" CACHE STRING "TAMM_CXX_FLAGS" FORCE)
        endif()            
    endif()
    
    bundle_cmake_args(DEPENDENCY_CMAKE_OPTIONS BLAS_INT4)

    print_banner("Locating Dependencies and Creating Targets")
    ################################################################################
    #
    # Add the subprojects, their dependencies, and their tests
    #
    ################################################################################

    foreach(__project ${NWX_PROJECTS})
        foreach(depend ${${__project}_DEPENDENCIES})
            find_or_build_dependency(${depend})
            are_we_building(${depend} were_building)
            if(were_building)
                list(APPEND DEPENDS_WERE_BUILDING ${depend})
            else()
                list(APPEND DEPENDS_WE_FOUND ${depend})
                package_dependency(${depend} DEPENDENCY_PATHS)
            endif()

        endforeach()

        set(${NWX_CXX_FLAGS} "${${NWX_CXX_FLAGS}} ${TAMM_CXX_FLAGS}")
        bundle_cmake_strings(CORE_CMAKE_STRINGS ${NWX_CXX_FLAGS})

        if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
            find_library(stdfs_LIBRARY 
                NAMES c++fs 
                PATHS ${CMAKE_CXX_IMPLICIT_LINK_DIRECTORIES} 
                DOC "LIBC++ FS Library" 
            )
        else()
            find_library(stdfs_LIBRARY 
                NAMES stdc++fs 
                PATHS ${CMAKE_CXX_IMPLICIT_LINK_DIRECTORIES} 
                DOC "GNU FS Library" 
            )
        endif()
        message(STATUS "STDFS LIB: ${stdfs_LIBRARY}")
        if(stdfs_LIBRARY)
            list(APPEND TAMM_EXTRA_LIBS ${stdfs_LIBRARY})
        endif()

        if(TAMM_EXTRA_LIBS)
            bundle_cmake_strings(CORE_CMAKE_STRINGS TAMM_EXTRA_LIBS)
            message(STATUS "TAMM_EXTRA_LIBS: ${TAMM_EXTRA_LIBS}")
        endif()

        if(NWX_CUDA)
            bundle_cmake_strings(CORE_CMAKE_STRINGS NWX_CUDA NWX_GPU_ARCH)
        endif()

        ExternalProject_Add(${__project}_External
                SOURCE_DIR ${${__project}_SRC_DIR}
                CMAKE_ARGS -DNWX_DEBUG_CMAKE=${NWX_DEBUG_CMAKE}
                           -DNWX_INCLUDE_DIR=${${__project}_INCLUDE_DIR}
                           ${CORE_CMAKE_OPTIONS}
                BUILD_ALWAYS 1
                INSTALL_COMMAND ${CMAKE_MAKE_PROGRAM} install DESTDIR=${STAGE_DIR}
                CMAKE_CACHE_ARGS ${CORE_CMAKE_LISTS}
                                 ${CORE_CMAKE_STRINGS}
                                 ${DEPENDENCY_PATHS}
                -DNWX_DEPENDENCIES:STRING=${${__project}_DEPENDENCIES}
                )

        foreach(depend ${${__project}_DEPENDENCIES})
            add_dependencies(${__project}_External ${depend}_External)
        endforeach()

        if(${BUILD_TESTS})
            list(APPEND TEST_DEPENDS "CMakeBuild" "${__project}")
            ExternalProject_Add(${__project}_Tests_External
                    SOURCE_DIR ${${__project}_TEST_DIR}
                    CMAKE_ARGS -DSUPER_PROJECT_ROOT=${SUPER_PROJECT_ROOT}
                               -DNWX_DEBUG_CMAKE=${NWX_DEBUG_CMAKE}
                               -DSTAGE_INSTALL_DIR=${STAGE_INSTALL_DIR}
                               ${CORE_CMAKE_OPTIONS}

                    BUILD_ALWAYS 1
                    INSTALL_COMMAND ${CMAKE_MAKE_PROGRAM} install DESTDIR=${TEST_STAGE_DIR}
                    CMAKE_CACHE_ARGS ${CORE_CMAKE_LISTS}
                                     ${CORE_CMAKE_STRINGS}
                                     ${DEPENDENCY_PATHS}
                                     -DNWX_DEPENDENCIES:LIST=${TEST_DEPENDS}
                    )
            add_dependencies(${__project}_Tests_External ${__project}_External)

            # This file will allow us to run ctest in the top-level build dir
            # Basically it just defers to the actual top-level CTestTestfile.cmake in the
            # build directory for this project
            file(WRITE ${CMAKE_BINARY_DIR}/CTestTestfile.cmake
                    "subdirs(test_stage${CMAKE_INSTALL_PREFIX}/tests)")
        endif()
    endforeach()

    # Install the staging directory
    install(DIRECTORY ${STAGE_INSTALL_DIR}/
            DESTINATION ${CMAKE_INSTALL_PREFIX} USE_SOURCE_PERMISSIONS)

    ############################################################################
    #
    # Let the user know all the settings we worked out
    #
    ############################################################################

    print_banner("Summary of ${PROJECT_NAME} Configuration Settings:")
    message(STATUS "Found the following dependencies: ")
    foreach(__depend ${DEPENDS_WE_FOUND})
        message(STATUS "    ${__depend}")
    endforeach()
    message(STATUS "Will build the following dependencies: ")
    foreach(__depend ${DEPENDS_WERE_BUILDING})
        message(STATUS "    ${__depend}")
    endforeach()

    ############################################################################
    #
    # Make an uninstall target
    #
    ############################################################################
    configure_file(
        "${NWXBASE_CMAKE}/cmake_uninstall.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
        IMMEDIATE @ONLY)

        add_custom_target(uninstall
                COMMAND ${CMAKE_COMMAND} -P
                ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake)
endfunction()

