add_subdirectory(powerSpectrum)

if (FFTW_MPI_LIB AND USE_PNETCDF)
  add_subdirectory(readSlice)
endif()

