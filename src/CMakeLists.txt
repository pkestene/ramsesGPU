add_subdirectory(utils)

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
