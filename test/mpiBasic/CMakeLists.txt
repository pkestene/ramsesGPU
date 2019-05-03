#######################################################
add_executable(testMpiTopology_2D_C_API
  testMpiTopology_2D_C_API.cpp
  )
target_link_libraries     (testMpiTopology_2D_C_API PUBLIC MPI::MPI_CXX)

# add_test(NAME testMpiTopology_2D_C_API COMMAND mpirun -np 12 testMpiTopology_2D_C_API)

#######################################################
add_executable(testMpiTopology_3D_C_API 
  testMpiTopology_3D_C_API.cpp
  )
target_link_libraries     (testMpiTopology_3D_C_API PUBLIC MPI::MPI_CXX)
# add_test(NAME testMpiTopology_3D_C_API COMMAND mpirun -np 27 testMpiTopology_3D_C_API)

#######################################################
add_executable(testMpiTopology_2D
  testMpiTopology_2D.cpp
  )
target_link_libraries(testMpiTopology_2D
  PRIVATE
  RamsesGPU::mpiUtils)

#######################################################
add_executable(testMpiHelloGlobalSession
  testMpiHelloGlobalSession.cpp
  )
target_link_libraries(testMpiHelloGlobalSession
  PRIVATE
  RamsesGPU::mpiUtils)

if (USE_VTK)
  add_executable(testVtkXMLPImageDataWriter
    testVtkXMLPImageDataWriter.cpp)
  
  if ("${VTK_MAJOR_VERSION}" GREATER 5)
    target_compile_options(testVtkXMLPImageDataWriter PUBLIC -DHAVE_VTK6)
  endif()

  target_link_libraries(testVtkXMLPImageDataWriter ${VTK_LIBRARIES})
endif()
