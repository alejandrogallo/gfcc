if(CMAKE_CXX_COMPILER_ID STREQUAL "XL"
    OR CMAKE_CXX_COMPILER_ID STREQUAL "Cray"
    OR CMAKE_CXX_COMPILER_ID STREQUAL "MSVC"
    OR CMAKE_CXX_COMPILER_ID STREQUAL "Intel" 
    OR CMAKE_CXX_COMPILER_ID STREQUAL "PGI"
    OR CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
        message(FATAL_ERROR "TAMM cannot be currently built with ${CMAKE_CXX_COMPILER_ID} compilers.")
endif()

if("${CMAKE_HOST_SYSTEM_NAME}" STREQUAL "Darwin")
    if (USE_CUDA)
        message(FATAL_ERROR "TAMM does not support building with GPU support \
        on MACOSX. Please use USE_CUDA=OFF for MACOSX builds.")
    endif()
    
    if(CMAKE_CXX_COMPILER_ID STREQUAL "Intel" 
        OR CMAKE_CXX_COMPILER_ID STREQUAL "PGI")
        message(FATAL_ERROR "TAMM does not support ${CMAKE_CXX_COMPILER_ID} compilers on MACOSX.")
    endif()
endif()

macro(get_compiler_exec_name comp_exec_path)
    get_filename_component(comp_exec_name ${comp_exec_path} NAME_WE)
endmacro()

macro(check_compiler_version lang_arg comp_type comp_version)
    if(CMAKE_${lang_arg}_COMPILER_ID STREQUAL "${comp_type}")
        if(CMAKE_${lang_arg}_COMPILER_VERSION VERSION_LESS "${comp_version}")
            get_compiler_exec_name("${CMAKE_${lang_arg}_COMPILER}")
            message(FATAL_ERROR "${comp_exec_name} version provided (${CMAKE_${lang_arg}_COMPILER_VERSION}) \
            is insufficient. Need ${comp_exec_name} >= ${comp_version} for building TAMM.")
        endif()
    endif()
endmacro()

set(ARMCI_NETWORK_TAMM OPENIB MPI-PR MPI-TS)
if(DEFINED ARMCI_NETWORK)
    list(FIND ARMCI_NETWORK_TAMM ${ARMCI_NETWORK} _index)
    if(${_index} EQUAL -1)
        message(FATAL_ERROR "TAMM only supports building GA using one of ${ARMCI_NETWORK_TAMM}, default is MPI-PR")
    endif()
endif()

check_compiler_version(C Clang 7)
check_compiler_version(CXX Clang 7)

check_compiler_version(C GNU 8.1)
check_compiler_version(CXX GNU 8.1)
check_compiler_version(Fortran GNU 8.1)

#TODO:Check for GCC>=8 compatibility
# check_compiler_version(C Intel 19)
# check_compiler_version(CXX Intel 19)
# check_compiler_version(Fortran Intel 19)

#TODO:Check for GCC>=8 compatibility
check_compiler_version(C PGI 18)
check_compiler_version(CXX PGI 18)
check_compiler_version(Fortran PGI 18)

if(USE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
            
        set(OUTPUTFILE ${CMAKE_CURRENT_BINARY_DIR}/cuda_arch_detect_script) 
        set(CUDAFILE ${CMAKE_CURRENT_SOURCE_DIR}/cmake/cuda_arch_detect.cu)
        execute_process(COMMAND nvcc ${CUDA_CUDART_LIBRARY} ${CUDAFILE} -o ${OUTPUTFILE})
        execute_process(COMMAND ${OUTPUTFILE}
                        RESULT_VARIABLE CUDA_RETURN_CODE
                        OUTPUT_VARIABLE ARCH)
        if(${CUDA_RETURN_CODE} EQUAL 0)
            message(STATUS "CUDA Architecture: ${ARCH}")     
            set(NV_GPU_ARCH ${ARCH})                   
        else()
            message(WARNING "Setting CUDA Architecture to: 35")  
            set(NV_GPU_ARCH 35)
        endif()
    else()
        if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER "8.5")
          get_compiler_exec_name("${CMAKE_CXX_COMPILER}")
          message(FATAL_ERROR "${comp_exec_name} version provided (${CMAKE_CXX_COMPILER_VERSION}) \
    is not supported by CUDA version provided. Need ${comp_exec_name} = 8.x for building TAMM with GPU support.")
        endif()
       message(FATAL_ERROR "CUDA Toolkit not found.")
    endif()
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 10.1)
        message(FATAL_ERROR "CUDA version provided \
         (${CMAKE_CUDA_COMPILER_VERSION}) \
         is insufficient. Need CUDA >= 10.1)")
    endif()
    
endif()


