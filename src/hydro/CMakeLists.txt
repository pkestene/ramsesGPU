add_library(hydro "")

target_sources(hydro
  PRIVATE
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
    PRIVATE
    mhd_godunov_unsplit_cpu_v0.cpp
    mhd_godunov_unsplit_cpu_v1.cpp
    mhd_godunov_unsplit_cpu_v2.cpp
    mhd_godunov_unsplit_cpu_v3.cpp
    )
endif(NOT USE_CUDA)

if (USE_MPI)
  target_sources(hydro
    PRIVATE
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
      PRIVATE
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

if (USE_PNETCDF)
  target_compile_options(hydro
    PUBLIC
    -DUSE_PNETCDF)
endif(USE_PNETCDF)

if (USE_NETCDF4)
  target_compile_options(hydro
    PUBLIC
    -DUSE_NETCDF4)
endif(USE_NETCDF4)

if (USE_TIMING)
  target_compile_options(hydro
    PUBLIC
    -DDO_TIMING)
endif(USE_TIMING)

target_include_directories(hydro
  PUBLIC
  ${CMAKE_SOURCE_DIR}/src)

target_link_libraries(hydro
  PUBLIC
  RamsesGPU::cnpy
  RamsesGPU::config)

if(USE_MPI)
  target_link_libraries(hydro
    PUBLIC
    RamsesGPU::mpiUtils)
endif(USE_MPI)

if (USE_HDF5)
  target_link_libraries(hydro
    PUBLIC
    hdf5::hdf5)
endif(USE_HDF5)

if (USE_PNETCDF)
  target_link_libraries(hydro
    PUBLIC
    RamsesGPU::pnetcdf)
endif(USE_PNETCDF)

if (USE_VTK)
  target_link_libraries(hydro
    PUBLIC
    ${VTK_LIBRARIES})
endif(USE_VTK)

add_library(RamsesGPU::hydro ALIAS hydro)
