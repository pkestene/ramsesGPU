############################################################
add_executable(testCopyBorderBuffer "")
target_sources(testCopyBorderBuffer
  PUBLIC
  testCopyBorderBuffer.cpp)
target_include_directories(testCopyBorderBuffer
  PUBLIC
  ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(testCopyBorderBuffer
  PUBLIC
  RamsesGPU::hydro)

############################################################
add_executable(testCopyBorderBufferShear "")
target_sources(testCopyBorderBufferShear
  PUBLIC
  testCopyBorderBufferShear.cpp)
target_include_directories(testCopyBorderBufferShear
  PUBLIC
  ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(testCopyBorderBufferShear
  PUBLIC
  RamsesGPU::hydro)

############################################################
if (USE_MPI)
  ############################################################
  add_executable(testHydroMpiParameters "")
  target_sources(testHydroMpiParameters PUBLIC
    testHydroMpiParameters.cpp)
  target_link_libraries(testHydroMpiParameters PUBLIC
    RamsesGPU::config
    RamsesGPU::hydro
    RamsesGPU::mpiUtils)

  ############################################################
  add_executable(testMpiBuffer "")
  target_sources(testMpiBuffer PUBLIC
    testMpiBuffer.cpp)
  target_link_libraries(testMpiBuffer PUBLIC
    RamsesGPU::config
    RamsesGPU::hydro
    RamsesGPU::mpiUtils)
    
  ############################################################
  add_executable(testMpiOutputVtk "")
  target_sources(testMpiOutputVtk PUBLIC
    testMpiOutputVtk.cpp)
  if (USE_CUDA)
    set_source_files_properties(testMpiOutputVtk.cpp PROPERTIES LANGUAGE CUDA)
  endif(USE_CUDA)
  target_link_libraries(testMpiOutputVtk PUBLIC
    RamsesGPU::hydro
    RamsesGPU::config
    RamsesGPU::cnpy
    RamsesGPU::monitoring
    RamsesGPU::mpiUtils)
    
  ############################################################
  if (NOT USE_CUDA)
    add_executable(testHydroRunBaseMpi_border "")
    target_sources(testHydroRunBaseMpi_border PUBLIC
      testHydroRunBaseMpi_border.cpp)
    target_link_libraries(testHydroRunBaseMpi_border PUBLIC
      RamsesGPU::hydro
      RamsesGPU::config
      RamsesGPU::cnpy
      RamsesGPU::monitoring
      RamsesGPU::mpiUtils)
  endif(NOT USE_CUDA)
endif(USE_MPI)
