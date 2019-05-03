set(HYDRO_NO_MPI_SRC
  ${CMAKE_CURRENT_SOURCE_DIR}/gpu_macros.cpp 
  ${CMAKE_CURRENT_SOURCE_DIR}/constants.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/turbulenceInit.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/structureFunctions.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/initHydro.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/RandomGen.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/Forcing_OrnsteinUhlenbeck.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/HydroRunBase.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/HydroRunGodunov.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/HydroRunKT.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/HydroRunRelaxingTVD.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/MHDRunBase.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/MHDRunGodunov.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/HydroRunGodunovZslab.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/MHDRunGodunovZslab.cpp)

if (NOT USE_CUDA)
  set(HYDRO_NO_MPI_SRC ${HYDRO_NO_MPI_SRC}
    ${CMAKE_CURRENT_SOURCE_DIR}/mhd_godunov_unsplit_cpu_v0.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/mhd_godunov_unsplit_cpu_v1.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/mhd_godunov_unsplit_cpu_v2.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/mhd_godunov_unsplit_cpu_v3.cpp
    )
endif(NOT USE_CUDA)

set(HYDRO_MPI_SRC
  ${CMAKE_CURRENT_SOURCE_DIR}/HydroMpiParameters.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/HydroRunBaseMpi.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/HydroRunGodunovMpi.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/MHDRunGodunovMpi.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/HydroRunGodunovZslabMpi.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/MHDRunGodunovZslabMpi.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/structureFunctionsMpi.cpp 
  )

if (NOT USE_CUDA)
  set(HYDRO_MPI_SRC ${HYDRO_MPI_SRC}
    ${CMAKE_CURRENT_SOURCE_DIR}/mhd_godunov_mpi_unsplit_cpu_v0.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/mhd_godunov_mpi_unsplit_cpu_v1.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/mhd_godunov_mpi_unsplit_cpu_v2.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/mhd_godunov_mpi_unsplit_cpu_v3.cpp)
endif(NOT USE_CUDA)


if (USE_MPI)
  set(HYDRO_SRC ${HYDRO_NO_MPI_SRC} ${HYDRO_MPI_SRC})
else(USE_MPI)
  set(HYDRO_SRC ${HYDRO_NO_MPI_SRC})
endif(USE_MPI)

if(USE_CUDA)
  foreach(file ${HYDRO_SRC})
    set_source_files_properties(${file} PROPERTIES LANGUAGE CUDA)
  endforeach()
endif(USE_CUDA)

########################################################################
add_library(hydro "")
target_sources(hydro
  PUBLIC
  ${HYDRO_SRC})

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
