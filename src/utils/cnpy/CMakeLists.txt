add_library(cnpy SHARED
  cnpy.cpp zlib/crc32.c)
target_include_directories(cnpy
  PUBLIC
  ${CMAKE_SOURCE_DIR} )

add_executable(cnpy_example cnpy_example.cpp)
target_include_directories(cnpy_example PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(cnpy_example cnpy)
