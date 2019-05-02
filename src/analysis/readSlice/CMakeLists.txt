#
# read slice of pnetcdf data file
#

add_executable(readSlice "")
target_sources(readSlice
  PUBLIC
  readSlice.cpp)
target_compile_options(readSlice 
  PUBLIC
  -DUSE_FFTW3_MPI -DUSE_PNETCDF)
target_include_directories(readSlice 
  PUBLIC
  ${PNETCDF_INCLUDE_DIRS}
  ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(readSlice
  PUBLIC
  RamsesGPU::config
  RamsesGPU::cnpy
  RamsesGPU::mpiUtils
  ${PNETCDF_LIBRARIES})
