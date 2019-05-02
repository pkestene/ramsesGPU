add_library(config "")

target_sources(config
  PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/ConfigMap.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ConfigMap.h
  ${CMAKE_CURRENT_SOURCE_DIR}/inih/ini.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/inih/ini.h
  ${CMAKE_CURRENT_SOURCE_DIR}/inih/INIReader.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/inih/INIReader.h
  )

target_include_directories(config
  PUBLIC
  ${CMAKE_SOURCE_DIR}/src
  ${CMAKE_SOURCE_DIR}/src/utils
  ${CMAKE_CURRENT_SOURCE_DIR}/inih ) 

if (USE_MPI)
  target_link_libraries(config PUBLIC MPI::MPI_CXX)
endif(USE_MPI)

#Add an alias so that library can be used inside the build tree
add_library(RamsesGPU::config ALIAS config)
