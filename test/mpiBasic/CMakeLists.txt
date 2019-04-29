#######################################################
add_executable(testMpiTopology_2D_C_API
  testMpiTopology_2D_C_API.cpp
  )
# add_test(NAME testMpiTopology_2D_C_API COMMAND mpirun -np 12 testMpiTopology_2D_C_API)

#######################################################
add_executable(testMpiTopology_3D_C_API 
  testMpiTopology_3D_C_API.cpp
  )
# add_test(NAME testMpiTopology_3D_C_API COMMAND mpirun -np 27 testMpiTopology_3D_C_API)

#######################################################
add_executable(testMpiTopology_2D
  testMpiTopology_2D.cpp
  )
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
