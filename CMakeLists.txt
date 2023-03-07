cmake_minimum_required(VERSION 3.18)
project(RamsesGPU LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS OFF)

# custom cmake macros location
list(INSERT CMAKE_MODULE_PATH 0 "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
list(INSERT CMAKE_MODULE_PATH 0 "${CMAKE_CURRENT_SOURCE_DIR}/cmake/cuda")

# Prevent from build in source tree
include(preventBuildInSource)

# Init build type: Release, Debug, ...
include(initBuildType)

# useful for debugging cmake
include(print_target_properties)

# always export compile commands database
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


####################################################################
# other options
####################################################################
option (USE_CUDA "build with CUDA support" OFF)
option (USE_DOUBLE "build with double precision" ON)
option (USE_HDF5 "build HDF5 input/output support" OFF)
option (USE_PNETCDF "build PNETCDF input/output support (MPI required)" OFF)
option (USE_VTK "Build VTK from system library (not hand written VTK)" OFF)
option (USE_MPI "Activate MPI build" OFF)
option (USE_MPI_CUDA_AWARE_ENFORCED "Some MPI cuda-aware implementation are not well detected; use this to enforce" OFF)
option (USE_FPE_DEBUG "build with floating point Nan tracing (signal handler)" OFF)
option (USE_QT "build some test demo gui application with Qt" OFF)
option (USE_GLUT "build some test demo gui application with GLUT" OFF)
option (USE_TIMING "Do fine level timing measurements" OFF)

#####################################################################
# External packages: MPI, ...
#####################################################################

#####################################################################
if (USE_CUDA)
  enable_language(CUDA)

  message("Using CUDAToolkit macros")
  find_package(CUDAToolkit REQUIRED)

  if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  endif()

  # we also need extra stuff to make sure compile flags are correctly
  # passed to nvcc / host compiler
  include(protect_nvcc_flags)
  include(protect_pthread_flag)

endif()

# Enable MODULE mode (via pkg-config) to find packages
# see https://cmake.org/cmake/help/latest/command/find_package.html
find_package(PkgConfig REQUIRED)

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

    ##################################################################
    # if CUDA is enabled we need to prevent flag "-pthread" to be passed
    # to nvcc compiler
    # this is probably done in a very recent version of cmake
    # see https://gitlab.kitware.com/cmake/cmake/commit/e2d54e5c40c517698c856b6802b51e0c09d5dca6
    #
    # When this is consumed for compiling CUDA, use '-Xcompiler' to wrap '-pthread'.
    ###string(REPLACE "-pthread" "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler >-pthread"
    ###  _MPI_CXX_COMPILE_OPTIONS "${MPI_CXX_COMPILE_OPTIONS}")

    # To be analyzed if the following (dirty) trick is still needed with
    # more recent version of cmake
    string(REPLACE "-pthread" $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler >-pthread  _MPI_CXX_COMPILE_OPTIONS "${MPI_CXX_COMPILE_OPTIONS}")

    #message("DEBUG : [before] ${MPI_CXX_COMPILE_OPTIONS} ---- [after] ${_MPI_CXX_COMPILE_OPTIONS}")

    set_property(TARGET MPI::MPI_CXX PROPERTY INTERFACE_COMPILE_OPTIONS "${_MPI_CXX_COMPILE_OPTIONS}")
    unset(_MPI_CXX_COMPILE_OPTIONS)


    ##################################################################

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

    # make sure to use Pnetcdf version >= 1.9 built as a shared library
    #find_package(PNETCDF REQUIRED)

    pkg_check_modules(RAMSESGPU_PNETCDF QUIET IMPORTED_TARGET pnetcdf)

    if(RAMSESGPU_PNETCDF_FOUND)
      message("PnetCDF found via pkg-config")
      set(PNETCDF_FOUND TRUE)
      add_library(RamsesGPU::pnetcdf ALIAS PkgConfig::RAMSESGPU_PNETCDF)
      add_compile_options(-DUSE_PNETCDF)
    else()
      message(WARNING "PnetCDF not found !")
    endif()

    # if (PNETCDF_FOUND)
    #   add_compile_options(-DUSE_PNETCDF)
    # endif(PNETCDF_FOUND)

  endif(USE_PNETCDF)

else (USE_MPI)

  if (USE_PNETCDF)
    message("You can use PnetCDF only if MPI is acticated !")
  endif(USE_PNETCDF)

endif (USE_MPI)

#####################################################################
# VTK from system
#####################################################################
if (USE_VTK)
  # the minimal component list is evaluated thanks to command:
  # ${VTK_SOURCES}/Utilities/Maintenance/WhatModulesVTK.py ${VTK_SOURCES} ${MY_APP}
  find_package(VTK COMPONENTS
    vtkCommonCore
    vtkCommonDataModel
    vtkCommonExecutionModel
    vtkCommonMath
    vtkCommonMisc
    vtkCommonSystem
    vtkCommonTransforms
    vtkIOCore
    vtkIOGeometry
    vtkIOImage
    vtkIOLegacy
    vtkIOXML
    vtkIOXMLParser
    vtkParallelCore
    vtkParallelMPI
    vtkIOParallelXML
    )
  include(${VTK_USE_FILE})
  message("VTK version: ${VTK_MAJOR_VERSION}.${VTK_MINOR_VERSION}")
endif(USE_VTK)

#####################################################################
# PAPI for performance counters
#####################################################################
find_package(PAPI)
if (NOT PAPI_FOUND)
  set(PAPI_INCLUDE_DIR "")
  set(PAPI_LIBRARY "")
endif(NOT PAPI_FOUND)

#####################################################################
# FFTW
#####################################################################
pkg_check_modules(RAMSESGPU_FFTW3 QUIET IMPORTED_TARGET fftw3 fftw3f)
if(RAMSESGPU_FFTW3_FOUND)
  message("fftw found via pkg-config")
  add_library(RamsesGPU::fftw ALIAS PkgConfig::RAMSESGPU_FFTW3)
else()
  message(WARNING "Please adjust environment variable PKG_CONFIG_PATH to setup path to fftw3.pc.")
endif()


#####################################################################
# QT
#####################################################################
if (USE_QT)
  find_package(Qt5Widgets)
  find_package(Qt5OpenGL)
  find_package(OpenGL)

  if (Qt5Widgets_FOUND)
    message("Qt5Widgets found - ${Qt5Widgets_VERSION_STRING}")
  else()
    message("Qt5Widgets not found ! Please turn off USE_QT or install Qt5")
  endif()

  # Instruct CMake to run moc automatically when needed
  set(CMAKE_AUTOMOC ON)

  # Create code from a list of Qt designer ui files
  set(CMAKE_AUTOUIC ON)

endif(USE_QT)

#####################################################################
# GLUT
#####################################################################
if (USE_GLUT)

  find_package(GLUT)
  find_package(OpenGL)

  message("GLUT found")

endif(USE_GLUT)

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
message("  CMake version          : ${CMAKE_VERSION}")
if (NOT CMAKE_BUILD_TYPE)
  message("  CMake build type       : NOT SET !")
else()
  message("  CMake build type       : ${CMAKE_BUILD_TYPE}")
endif()
message("  CMake install prefix   : ${CMAKE_INSTALL_PREFIX}")
message("  CMake system processor : ${CMAKE_SYSTEM_PROCESSOR}")
message("  CMake system name (OS) : ${CMAKE_SYSTEM_NAME}")
message("")

message("  Cxx compiler Id      : ${CMAKE_CXX_COMPILER_ID}")
message("  Cxx compiler version : ${CMAKE_CXX_COMPILER_VERSION}")
message("  Cxx compiler exe     : ${CMAKE_CXX_COMPILER}")
message("  Cxx flags            : ${CMAKE_CXX_FLAGS}")
message("  Cxx compiler wrapper : ${CMAKE_CXX_COMPILER_WRAPPER}")
message("")

if (USE_MPI)
  message("  MPI enabled")
  message("    MPI found            : ${MPI_FOUND}")
  message("    MPI standard version : ${MPI_CXX_VERSION}")
  # note : make sure you correctly your C, C++ and Fortran compiler
  # through variable CC, CXX and FC
  message("    MPI_VENDOR           : ${MPI_VENDOR}")
  message("    MPI library version  : ${MPI_CXX_LIBRARY_VERSION_STRING}")
  message("    MPI fortran compiler : ${MPI_CXX_COMPILER}")
  message("    MPI headers          : ${MPI_CXX_INCLUDE_DIRS}")
  message("    MPI libraries        : ${MPI_CXX_LIBRARIES}")
else(USE_MPI)
  message("  MPI not enabled")
endif(USE_MPI)
message("")

if (USE_CUDA)
  message("  CUDA compiler ID      : ${CMAKE_CUDA_COMPILER_ID}")
  message("  CUDA compiler Version : ${CMAKE_CUDA_COMPILER_VERSION}")
  message("  CUDA Compiler         : ${CMAKE_CUDA_COMPILER}")
  message("  CUDA Compiler exec    : ${CUDA_NVCC_EXECUTABLE}")
  message("  CUDA Compile flags    : ${CMAKE_CUDA_FLAGS}")
  message("  CUDA toolkit found    : ${CUDAToolkit_FOUND}")
  message("  CUDA toolkit version  : ${CUDAToolkit_VERSION}")
  message("  CUDA toolkit nvcc     : ${CUDAToolkit_NVCC_EXECUTABLE}")
else(USE_CUDA)
  message("  CUDA not enabled")
endif(USE_CUDA)
message("")

if (HDF5_FOUND)
  message("  HDF5 found version    : ${HDF5_VERSION}")
  message("  HDF5 definitions      : ${HDF5_DEFINITIONS}")
  message("  HDF5 parallel         : ${HDF5_IS_PARALLEL}")
  message("  HDF5 includes dirs    : ${HDF5_INCLUDE_DIRS}")
  message("  HDF5 libraries        : ${HDF5_LIBRARIES}")
endif(HDF5_FOUND)
message("")

if (RAMSESGPU_PNETCDF_FOUND)
  message("  PNETCDF found version : ${RAMSESGPU_PNETCDF_VERSION}")
  message("  PNETCDF include dirs  : ${RAMSESGPU_PNETCDF_CFLAGS}")
  message("  PNETCDF libraries     : ${RAMSESGPU_PNETCDF_LDFLAGS}")
endif(RAMSESGPU_PNETCDF_FOUND)
message("")
