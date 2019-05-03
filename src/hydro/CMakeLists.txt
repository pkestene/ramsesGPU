add_library(hydro "")

target_sources(hydro
  PUBLIC
  gpu_macros.cpp 
  constants.cpp
  turbulenceInit.cpp
  structureFunctions.cpp
  initHydro.cpp
  RandomGen.cpp
  Forcing_OrnsteinUhlenbeck.cpp
  HydroRunBase.cpp
  HydroRunGodunov.cpp
  HydroRunKT.cpp
  HydroRunRelaxingTVD.cpp
  MHDRunBase.cpp
  MHDRunGodunov.cpp
  HydroRunGodunovZslab.cpp
  MHDRunGodunovZslab.cpp)

if (NOT USE_CUDA)
  target_sources(hydro
    PUBLIC
    mhd_godunov_unsplit_cpu_v0.cpp
    mhd_godunov_unsplit_cpu_v1.cpp
    mhd_godunov_unsplit_cpu_v2.cpp
    mhd_godunov_unsplit_cpu_v3.cpp
    )
endif(NOT USE_CUDA)

if (USE_MPI)
  target_sources(hydro
    PUBLIC
    HydroMpiParameters.cpp
    HydroRunBaseMpi.cpp
    HydroRunGodunovMpi.cpp
    MHDRunGodunovMpi.cpp
    HydroRunGodunovZslabMpi.cpp
    MHDRunGodunovZslabMpi.cpp
    structureFunctionsMpi.cpp 
    )
  
  if (NOT USE_CUDA)
    target_sources(hydro
      PUBLIC
      mhd_godunov_mpi_unsplit_cpu_v0.cpp
      mhd_godunov_mpi_unsplit_cpu_v1.cpp
      mhd_godunov_mpi_unsplit_cpu_v2.cpp
      mhd_godunov_mpi_unsplit_cpu_v3.cpp)
  endif(NOT USE_CUDA)
endif(USE_MPI)

# make sure cpp files are recognized as cuda source files
# when building for GPU executable
if(USE_CUDA)

  get_property(HYDRO_SRC TARGET hydro PROPERTY SOURCES)

  foreach(file ${HYDRO_SRC})
    set_source_files_properties(${file} PROPERTIES LANGUAGE CUDA)
  endforeach()
endif(USE_CUDA)

########################################################################
if (USE_VTK)  
  target_compile_options(hydro
    PUBLIC
    -DUSE_VTK)
endif(USE_VTK)

if (USE_HDF5)  
  target_compile_options(hydro
    PUBLIC
    -DUSE_HDF5 -DH5_NO_DEPRECATED_SYMBOLS)
endif(USE_HDF5)

# TODO HDF5 parallel

if (USE_PNETCDF)
  target_compile_options(hydro
    PUBLIC
    -DUSE_PNETCDF)
  target_include_directories(hydro
    PUBLIC
    ${PNETCDF_INCLUDE_DIRS})
endif(USE_PNETCDF)

if (USE_NETCDF4)  
  target_compile_options(hydro
    PUBLIC
    -DUSE_NETCDF4)
endif(USE_NETCDF4)

target_include_directories(hydro
  PUBLIC
  ${CMAKE_SOURCE_DIR}/src)

if (USE_HDF5)
  target_include_directories(hydro
  PUBLIC
  ${HDF5_INCLUDE_DIRS})
endif(USE_HDF5)

target_link_libraries(hydro
  PUBLIC
  RamsesGPU::config)

if(USE_MPI)
  target_link_libraries(hydro
    PUBLIC
    RamsesGPU::mpiUtils)
endif(USE_MPI)

if (USE_HDF5)
  target_link_libraries(hydro
  PUBLIC
  ${HDF5_LIBRARIES})
endif(USE_HDF5)

if (USE_PNETCDF)
  target_link_libraries(hydro
  PUBLIC
  ${PNETCDF_LIBRARIES})
endif(USE_PNETCDF)

add_library(RamsesGPU::hydro ALIAS hydro)
