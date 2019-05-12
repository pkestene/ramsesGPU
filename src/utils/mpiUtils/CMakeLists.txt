add_library(mpiUtils "")

target_sources (mpiUtils
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/GlobalMpiSession.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/TestForException.h
  ${CMAKE_CURRENT_SOURCE_DIR}/ErrorPolling.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ErrorPolling.h
  ${CMAKE_CURRENT_SOURCE_DIR}/common_config.h
  ${CMAKE_CURRENT_SOURCE_DIR}/MpiComm.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/MpiCommCart.cpp
  PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/GlobalMpiSession.h
  ${CMAKE_CURRENT_SOURCE_DIR}/mpiEnums.h
  ${CMAKE_CURRENT_SOURCE_DIR}/MpiComm.h
  ${CMAKE_CURRENT_SOURCE_DIR}/MpiCommCart.h
  )
#Set target properties
target_include_directories(mpiUtils
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/
)
target_link_libraries (mpiUtils PUBLIC MPI::MPI_CXX)

#Add an alias so that library can be used inside the build tree
add_library(RamsesGPU::mpiUtils ALIAS mpiUtils)
