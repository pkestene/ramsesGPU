if(USE_CUDA)
  add_executable(testHelloMpiCuda "")

  target_sources(testHelloMpiCuda 
    PRIVATE
    testHelloMpiCuda.cu)

  # target_compile_options    (testHelloMpiCuda 
  #   PUBLIC 
  #   ${MPI_CXX_COMPILE_FLAGS})
  target_include_directories(testHelloMpiCuda 
    PUBLIC 
    ${CMAKE_SOURCE_DIR}/src/utils/mpiUtils)
  target_link_libraries(testHelloMpiCuda PUBLIC mpiUtils)

  # convert flags to something nvcc can process
  #cmake_cuda_convert_flags(INTERFACE_TARGET testHelloMpiCuda)
  
endif(USE_CUDA)
