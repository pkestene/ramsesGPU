add_library(cnpy SHARED
  cnpy.cpp zlib/crc32.c)
target_include_directories(cnpy
  PUBLIC
  ${CMAKE_SOURCE_DIR} )

add_library(RamsesGPU::cnpy ALIAS cnpy)

# test example
add_executable(cnpy_example cnpy_example.cpp)
target_link_libraries(cnpy_example 
  PUBLIC
  RamsesGPU::cnpy)
