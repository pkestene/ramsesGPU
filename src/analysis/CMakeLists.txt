add_subdirectory(powerSpectrum)

if (USE_MPI AND USE_PNETCDF)
  add_subdirectory(readSlice)
endif()

if (USE_MPI AND USE_PNETCDF AND NOT CUDA)
  add_subdirectory(structureFunctions)
endif()
