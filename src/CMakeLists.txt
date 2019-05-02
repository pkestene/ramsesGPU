add_subdirectory(utils)

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
