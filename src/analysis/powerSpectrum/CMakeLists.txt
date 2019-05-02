configure_file(fft.ini fft.ini)
configure_file(fBm.ini fBm.ini)

if (USE_MPI)

  #
  # ComputePowerSpectrumMpi
  #

  # only add this test if fftw3_mpi library is available
  if (FFTW_MPI_LIB AND USE_PNETCDF)
    add_executable(computePowerSpectrumMpi "")
    target_sources(computePowerSpectrumMpi
    PUBLIC
    fft_mpi.cpp
    pnetcdf_io.cpp
    powerSpectrumMpi.cpp)
  target_compile_options(computePowerSpectrumMpi 
    PUBLIC
    -DUSE_FFTW3_MPI -DUSE_PNETCDF)
  target_include_directories(computePowerSpectrumMpi 
    PUBLIC
    ${PNETCDF_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/src)
  target_link_libraries(computePowerSpectrumMpi
  PUBLIC
  RamsesGPU::config
  RamsesGPU::cnpy
  fftw3
  fftw3_mpi
  ${PNETCDF_LIBRARIES})

  endif()

  #
  # generate_fBm
  #
  if (FFTW_MPI_LIB AND USE_PNETCDF)
    add_executable(generate_fBm "")
    target_sources(generate_fBm
    PUBLIC
    fft_mpi.cpp
    pnetcdf_io.cpp
    generate_fBm.cpp)
  target_compile_options(generate_fBm 
    PUBLIC
    -DUSE_FFTW3_MPI -DUSE_PNETCDF -DGEN_FBM)
  target_include_directories(generate_fBm 
    PUBLIC
    ${PNETCDF_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/src)
  target_link_libraries(generate_fBm
  PUBLIC
  RamsesGPU::config
  RamsesGPU::cnpy
  fftw3
  fftw3_mpi
  ${PNETCDF_LIBRARIES})

  endif()

endif(USE_MPI)
