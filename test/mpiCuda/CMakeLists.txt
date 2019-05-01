if(USE_CUDA)
  add_executable(testHelloMpiCuda "")

  target_sources(testHelloMpiCuda 
    PRIVATE
    testHelloMpiCuda.cu)

  target_link_libraries(testHelloMpiCuda PUBLIC RamsesGPU::mpiUtils)

  # convert flags to something nvcc can process
  #cmake_cuda_convert_flags(INTERFACE_TARGET testHelloMpiCuda)
  
endif(USE_CUDA)
