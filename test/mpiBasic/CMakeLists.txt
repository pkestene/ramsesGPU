#######################################################
add_executable(testMpiTopology_2D_C_API
  testMpiTopology_2D_C_API.cpp
  )
target_compile_options    (testMpiTopology_2D_C_API PUBLIC ${MPI_CXX_COMPILE_FLAGS})
target_include_directories(testMpiTopology_2D_C_API PUBLIC ${MPI_CXX_INCLUDE_PATH})
target_link_libraries     (testMpiTopology_2D_C_API PUBLIC ${MPI_CXX_LIBRARIES})

# add_test(NAME testMpiTopology_2D_C_API COMMAND mpirun -np 12 testMpiTopology_2D_C_API)

#######################################################
add_executable(testMpiTopology_3D_C_API 
  testMpiTopology_3D_C_API.cpp
  )
target_compile_options    (testMpiTopology_3D_C_API PUBLIC ${MPI_CXX_COMPILE_FLAGS})
target_include_directories(testMpiTopology_3D_C_API PUBLIC ${MPI_CXX_INCLUDE_PATH})
target_link_libraries     (testMpiTopology_3D_C_API PUBLIC ${MPI_CXX_LIBRARIES})
# add_test(NAME testMpiTopology_3D_C_API COMMAND mpirun -np 27 testMpiTopology_3D_C_API)

#######################################################
add_executable(testMpiTopology_2D
  testMpiTopology_2D.cpp
  )
target_compile_options    (testMpiTopology_2D PUBLIC ${MPI_CXX_COMPILE_FLAGS})
target_include_directories(testMpiTopology_2D
  PUBLIC
  ${CMAKE_SOURCE_DIR}/src/utils/mpiUtils)
target_link_libraries(testMpiTopology_2D
  PRIVATE
  mpiUtils)

#######################################################
add_executable(testMpiHelloGlobalSession
  testMpiHelloGlobalSession.cpp
  )
target_include_directories(testMpiHelloGlobalSession 
  PUBLIC 
  ${CMAKE_SOURCE_DIR}/src/utils/mpiUtils)
target_link_libraries(testMpiHelloGlobalSession
  PRIVATE
  mpiUtils)

if (USE_VTK)
  add_executable(testVtkXMLPImageDataWriter
    testVtkXMLPImageDataWriter.cpp)
  
  if ("${VTK_MAJOR_VERSION}" GREATER 5)
    target_compile_options(testVtkXMLPImageDataWriter PUBLIC -DHAVE_VTK6)
  endif()

  target_link_libraries(testVtkXMLPImageDataWriter ${VTK_LIBRARIES})
endif()
