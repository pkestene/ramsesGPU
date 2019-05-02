if(USE_CUDA)

  #############################################################
  add_executable(testHelloMpiCuda "")

  target_sources(testHelloMpiCuda 
    PRIVATE
    testHelloMpiCuda.cu)

  target_link_libraries(testHelloMpiCuda PUBLIC RamsesGPU::mpiUtils)

  # convert flags to something nvcc can process
  #cmake_cuda_convert_flags(INTERFACE_TARGET testHelloMpiCuda)

  #############################################################
  add_executable(testBorderBufferCuda "")

  target_sources(testBorderBufferCuda 
    PRIVATE
    ${CMAKE_SOURCE_DIR}/src/hydro/gpu_macros.cpp
    testBorderBufferCuda.cu)

  set_source_files_properties(${CMAKE_SOURCE_DIR}/src/hydro/gpu_macros.cpp PROPERTIES LANGUAGE CUDA)

  target_include_directories(testBorderBufferCuda
    PUBLIC
    ${CMAKE_SOURCE_DIR}/src)
  target_link_libraries(testBorderBufferCuda PUBLIC RamsesGPU::mpiUtils)

  #############################################################
  add_executable(testBorderBufferCuda2 "")

  target_sources(testBorderBufferCuda2
    PRIVATE
    ${CMAKE_SOURCE_DIR}/src/hydro/gpu_macros.cpp
    testBorderBufferCuda2.cu)

  #set_source_files_properties(${CMAKE_SOURCE_DIR}/src/hydro/gpu_macros.cpp PROPERTIES LANGUAGE CUDA)

  target_include_directories(testBorderBufferCuda2
    PUBLIC
    ${CMAKE_SOURCE_DIR}/src)
  target_link_libraries(testBorderBufferCuda2 PUBLIC RamsesGPU::mpiUtils)


endif(USE_CUDA)
