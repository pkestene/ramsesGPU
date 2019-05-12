add_library(cnpy "")

target_sources(cnpy
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/cnpy.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/zlib/crc32.c)

target_include_directories(cnpy
  PUBLIC
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/src/utils )

set_target_properties(cnpy PROPERTIES LINKER_LANGUAGE CXX)

add_library(RamsesGPU::cnpy ALIAS cnpy)

# test example
add_executable(cnpy_example cnpy_example.cpp)
target_link_libraries(cnpy_example 
  PUBLIC
  RamsesGPU::cnpy)
