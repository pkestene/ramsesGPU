configure_file(test_pnetcdf.ini test_pnetcdf.ini)
configure_file(strucFunc.ini strucFunc.ini)

add_executable(test_pnetcdf_write "")
target_sources(test_pnetcdf_write
  PUBLIC
  pnetcdf_io.cpp
  test_pnetcdf_write.cpp)
target_compile_options(test_pnetcdf_write
  PUBLIC
  -DUSE_PNETCDF)
target_include_directories(test_pnetcdf_write
  PUBLIC
  ${PNETCDF_INCLUDE_DIRS}
  ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(test_pnetcdf_write
  PUBLIC
  RamsesGPU::config
  RamsesGPU::cnpy
  ${PNETCDF_LIBRARIES})


add_executable(computeStructureFunctions "")
target_sources(computeStructureFunctions
  PUBLIC
  common_sf.cpp
  pnetcdf_io.cpp
  structureFunctionsMpi.cpp)
target_compile_options(computeStructureFunctions 
  PUBLIC
  -DUSE_PNETCDF)
target_include_directories(computeStructureFunctions 
  PUBLIC
  ${PNETCDF_INCLUDE_DIRS}
  ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(computeStructureFunctions
  PUBLIC
  RamsesGPU::config
  RamsesGPU::cnpy
  ${PNETCDF_LIBRARIES})


add_executable(computeStructureFunctions_v2 "")
target_sources(computeStructureFunctions_v2
  PUBLIC
  common_sf.cpp
  pnetcdf_io.cpp
  structureFunctionsMpi_v2.cpp)
target_compile_options(computeStructureFunctions_v2 
  PUBLIC
  -DUSE_PNETCDF)
target_include_directories(computeStructureFunctions_v2 
  PUBLIC
  ${PNETCDF_INCLUDE_DIRS}
  ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(computeStructureFunctions_v2
  PUBLIC
  RamsesGPU::config
  RamsesGPU::cnpy
  ${PNETCDF_LIBRARIES})


add_executable(computeStructureFunctions_v3 "")
target_sources(computeStructureFunctions_v3
  PUBLIC
  common_sf.cpp
  pnetcdf_io.cpp
  structureFunctionsMpi_v3.cpp)
target_compile_options(computeStructureFunctions_v3 
  PUBLIC
  -DUSE_PNETCDF)
target_include_directories(computeStructureFunctions_v3 
  PUBLIC
  ${PNETCDF_INCLUDE_DIRS}
  ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(computeStructureFunctions_v3
  PUBLIC
  RamsesGPU::config
  RamsesGPU::cnpy
  ${PNETCDF_LIBRARIES})
