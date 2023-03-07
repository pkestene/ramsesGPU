add_executable(makeConfigHydro makeConfigHydro.cpp)
target_compile_options(makeConfigHydro PRIVATE -I${CMAKE_SOURCE_DIR}/src/hydro)

configure_file(test_run.sh.in test_run.sh)
configure_file(computeL2relatif.py.in computeL2relatif.py)


if (USE_MPI)

  add_subdirectory(mpiBasic)
  add_subdirectory(mpiCuda)
  add_subdirectory(mpiHydro)
  #add_subdirectory(mpiIO)

  if (RAMSESGPU_FFTW3_FOUND)
    add_subdirectory(testPoisson)
  endif(RAMSESGPU_FFTW3_FOUND)

endif(USE_MPI)
