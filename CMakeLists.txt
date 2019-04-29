cmake_minimum_required(VERSION 3.10)
project(RamsesGPU LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS OFF)

# custom cmake macros location
list(INSERT CMAKE_MODULE_PATH 0 "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Prevent from build in source tree
include(preventBuildInSource)

# Init build type: Release, Debug, ...
include(initBuildType)

# always export compile commands database
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)  

# CUDA support
option (USE_CUDA "build with CUDA support" OFF)
if (USE_CUDA)
  enable_language(CUDA)
  
  if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  endif()
endif()

####################################################################
# other options
####################################################################
option (USE_DOUBLE "build with double precision" ON)
option (USE_HDF5 "build HDF5 input/output support" OFF)
option (USE_PNETCDF "build PNETCDF input/output support (MPI required)" OFF)
option (USE_MPI "Activate MPI build" OFF)
option (USE_MPI_CUDA_AWARE_ENFORCED "Some MPI cuda-aware implementation are not well detected; use this to enforce" OFF)
option (USE_FPE_DEBUG "build with floating point Nan tracing (signal handler)" OFF)

#####################################################################
# External packages: MPI, ...
#####################################################################

#####################################################################
find_package(MPI)
if (USE_MPI)
  if(MPI_CXX_FOUND)
    message(STATUS "MPI support found")
    message(STATUS "MPI compile flags: " ${MPI_CXX_COMPILE_FLAGS})
    message(STATUS "MPI include path: " ${MPI_CXX_INCLUDE_PATH})
    message(STATUS "MPI LINK flags path: " ${MPI_CXX_LINK_FLAGS})
    message(STATUS "MPI libraries: " ${MPI_CXX_LIBRARIES})
    #include_directories(SYSTEM ${MPI_CXX_INCLUDE_PATH})
    
    #set(CMAKE_EXE_LINKER_FLAGS ${MPI_CXX_LINK_FLAGS})
    find_program(OMPI_INFO
      NAMES ompi_info
      HINTS ${MPI_CXX_LIBRARIES}/../bin)
    
    # Full command line to probe if cuda support in MPI implementation is enabled
    # ompi_info --parsable --all | grep mpi_built_with_cuda_support:value
    if (OMPI_INFO)
      execute_process(COMMAND ${OMPI_INFO}
        OUTPUT_VARIABLE _output)
      if ( (_output MATCHES "smcuda") OR (USE_MPI_CUDA_AWARE_ENFORCED) )
        message(STATUS "Found OpenMPI with CUDA support built.")
      else()
        message(WARNING "OpenMPI found, but it is not built with CUDA support.")
	add_compile_options(-DMPI_CUDA_AWARE_OFF)
      endif()
    endif()
  else()
    message(WARNING "Not compiling with MPI. Suppress this warning with -DUSE_MPI=OFF")
    set(USE_MPI OFF)
  endif(MPI_CXX_FOUND)
endif(USE_MPI)

#####################################################################
# HDF5
#####################################################################
# prefer using parallel HDF5 when build with mpi
if (USE_MPI)
  set(HDF5_PREFER_PARALLEL TRUE)
endif(USE_MPI)

if (USE_HDF5)
  find_package(HDF5 REQUIRED)
  if(HDF5_FOUND)
    #include_directories(${HDF5_INCLUDE_DIRS})
    set(my_hdf5_libs hdf5 hdf5_cpp)
    add_compile_options(-DUSE_HDF5)
    if (HDF5_IS_PARALLEL)
      add_compile_options(-DUSE_HDF5_PARALLEL)
    endif()
  endif(HDF5_FOUND)
endif(USE_HDF5)

#####################################################################
# PNETCDF
#####################################################################
if (USE_MPI)
  if (USE_PNETCDF)
    find_package(PNETCDF REQUIRED)
    if (PNETCDF_FOUND)
      add_compile_options(-DUSE_PNETCDF)
      #include_directories(${PNETCDF_INCLUDE_DIRS})
    endif(PNETCDF_FOUND)
  endif(USE_PNETCDF)
else (USE_MPI)
  if (USE_PNETCDF)
    message("You can use PnetCDF only if MPI is acticated !")
  endif(USE_PNETCDF)
endif (USE_MPI)

#####################################################################
# PAPI for performance counters
#####################################################################
find_package(PAPI)
if (NOT PAPI_FOUND)
  set(PAPI_INCLUDE_DIR "")
  set(PAPI_LIBRARY "")
endif(NOT PAPI_FOUND)

#
# common flags
#

# double precision support
if (USE_DOUBLE)
  add_compile_options(-DUSE_DOUBLE)
endif()

if (USE_MPI)
  add_compile_options(-DUSE_MPI)
endif()
  
if (USE_FPE_DEBUG)
  add_compile_options(-DUSE_FPE_DEBUG)
endif()

##########################################################
##########################################################
##########################################################

# build src
add_subdirectory(src)

# build test
add_subdirectory(test)

# Testing
enable_testing()


#-------------------------------------------------------------------#
# Configuration summary
#-------------------------------------------------------------------#
message("//===================================================")
message("  ${PROJECT_NAME} build configuration:")
message("//===================================================")
message("  C++ Compiler : ${CMAKE_CXX_COMPILER_ID} " 
                         "${CMAKE_CXX_COMPILER_VERSION} " 
                         "${CMAKE_CXX_COMPILER_WRAPPER}")
message("    ${CMAKE_CXX_COMPILER}")
message("")
