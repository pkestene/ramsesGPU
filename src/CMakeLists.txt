add_subdirectory(utils)
add_subdirectory(analysis)
add_subdirectory(hydro)

#
# MAIN application
#

# set exe name
if(USE_MPI)
  if (USE_CUDA)
    set(RamsesGPU_main_exe ramsesGPU_mpi_cuda)
  else(USE_CUDA)
    set(RamsesGPU_main_exe ramsesGPU_mpi_cpu)
  endif(USE_CUDA)
else(USE_MPI)
  if (USE_CUDA)
    set(RamsesGPU_main_exe ramsesGPU_cuda)
  else(USE_CUDA)
    set(RamsesGPU_main_exe ramsesGPU_cpu)
  endif(USE_CUDA)
endif(USE_MPI)

# add source : there is a different main is MPI is enabled
add_executable(${RamsesGPU_main_exe} "")
if (USE_MPI)
  target_sources(${RamsesGPU_main_exe}
    PUBLIC
    euler_mpi_main.cpp)
else(USE_MPI)
  target_sources(${RamsesGPU_main_exe}
    PUBLIC
    euler_main.cpp)
endif(USE_MPI)

if (USE_CUDA)
  set_source_files_properties(euler_main.cpp     PROPERTIES LANGUAGE CUDA)
  set_source_files_properties(euler_mpi_main.cpp PROPERTIES LANGUAGE CUDA)
endif()


target_include_directories(${RamsesGPU_main_exe}
  PUBLIC
  ${CMAKE_SOURCE_DIR}/src)

target_link_libraries(${RamsesGPU_main_exe}
  PUBLIC
  RamsesGPU::hydro
  RamsesGPU::config
  RamsesGPU::cnpy
  RamsesGPU::monitoring)
if (USE_MPI)
  target_link_libraries(${RamsesGPU_main_exe}
    PUBLIC
    RamsesGPU::mpiUtils)
endif(USE_MPI)

######################################################################
# not really useful for now - TBC
if (USE_QT)
  add_subdirectory(qtGui)
endif(USE_QT)

######################################################################
# To be moved elsewhere

#
# testRiemannHLLD
#
# add_executable(testRiemannHLLD "")
# target_sources(testRiemannHLLD PUBLIC
#   testRiemannHLLD.cpp
#   hydro/gpu_macros.cpp
#   hydro/constants.cpp)
# target_include_directories(testRiemannHLLD
#   PUBLIC
#   ${CMAKE_SOURCE_DIR}/src
#   ${CMAKE_SOURCE_DIR}/src/hydro)
# target_link_libraries(testRiemannHLLD
#   PRIVATE
#   RamsesGPU::config)

# if (USE_CUDA)
#   set_source_files_properties(testRiemannHLLD.cpp PROPERTIES LANGUAGE CUDA)
#   set_source_files_properties(hydro/gpu_macros.cpp PROPERTIES LANGUAGE CUDA)
#   set_source_files_properties(hydro/constants.cpp PROPERTIES LANGUAGE CUDA)
# endif(USE_CUDA)

#
# testTrace
#
# add_executable(testTrace "")
# target_sources(testTrace PUBLIC
#   testTrace.cpp
#   hydro/gpu_macros.cpp
#   hydro/constants.cpp)
# target_include_directories(testTrace
#   PUBLIC
#   ${CMAKE_SOURCE_DIR}/src
#   ${CMAKE_SOURCE_DIR}/src/hydro)
# target_link_libraries(testTrace
#   PRIVATE
#   RamsesGPU::config)

# # warning: since source file gpu_macros and constants.cpp
# # have already been declared as CUDA file, we need to stick with it
# if (USE_CUDA)
#   set_source_files_properties(testTrace.cpp PROPERTIES LANGUAGE CUDA)
# endif(USE_CUDA)

#if (USE_CUDA)
#print_target_properties(testTrace)
#CUDA_PROTECT_PTHREAD_FLAG(testTrace)
#endif(USE_CUDA)
