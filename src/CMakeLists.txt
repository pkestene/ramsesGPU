add_subdirectory(utils)
add_subdirectory(analysis)
add_subdirectory(hydro)

#
# MAIN application
#

if(USE_MPI)
  if (USE_CUDA)
    add_executable(ramsesGPU_mpi_cuda "")
    target_sources(ramsesGPU_mpi_cuda 
      PUBLIC
      euler_mpi_main.cpp)
    target_include_directories(ramsesGPU_mpi_cuda
      PUBLIC
      ${CMAKE_SOURCE_DIR}/src)
    target_link_libraries(ramsesGPU_mpi_cuda
      PUBLIC
      RamsesGPU::config
      RamsesGPU::cnpy
      RamsesGPU::hydro
      RamsesGPU::monitoring
      RamsesGPU::mpiUtils)
    set_source_files_properties(euler_mpi_main.cpp PROPERTIES LANGUAGE CUDA)
  endif(USE_CUDA)
else(USE_MPI)
  if (USE_CUDA)
    add_executable(ramsesGPU_cuda "")
    target_sources(ramsesGPU_cuda 
      PUBLIC
      euler_main.cpp)
    target_include_directories(ramsesGPU_cuda
      PUBLIC
      ${CMAKE_SOURCE_DIR}/src)
    target_link_libraries(ramsesGPU_cuda
      PUBLIC
      RamsesGPU::config
      RamsesGPU::cnpy
      RamsesGPU::hydro
      RamsesGPU::monitoring)
    set_source_files_properties(euler_main.cpp PROPERTIES LANGUAGE CUDA)
  endif(USE_CUDA)  
endif(USE_MPI)


# not really useful
if (USE_QT)
  add_subdirectory(qtGui)
endif(USE_QT)

#
# testRiemannHLLD
#
add_executable(testRiemannHLLD "")
target_sources(testRiemannHLLD PUBLIC
  testRiemannHLLD.cpp
  hydro/gpu_macros.cpp 
  hydro/constants.cpp)
target_include_directories(testRiemannHLLD 
  PUBLIC
  ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_SOURCE_DIR}/src/hydro)
target_link_libraries(testRiemannHLLD
  PRIVATE
  RamsesGPU::config)

if (USE_CUDA)
  set_source_files_properties(testRiemannHLLD.cpp PROPERTIES LANGUAGE CUDA)
  set_source_files_properties(hydro/gpu_macros.cpp PROPERTIES LANGUAGE CUDA)
  set_source_files_properties(hydro/constants.cpp PROPERTIES LANGUAGE CUDA)
endif(USE_CUDA)

#
# testTrace
#
add_executable(testTrace "")
target_sources(testTrace PUBLIC
  testTrace.cpp
  hydro/gpu_macros.cpp 
  hydro/constants.cpp)
target_include_directories(testTrace 
  PUBLIC
  ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_SOURCE_DIR}/src/hydro)
target_link_libraries(testTrace
  PRIVATE
  RamsesGPU::config)

# warning: since source file gpu_macros and constants.cpp
# have already been declared as CUDA file, we need to stick with it
if (USE_CUDA)
  set_source_files_properties(testTrace.cpp PROPERTIES LANGUAGE CUDA)
endif(USE_CUDA)
