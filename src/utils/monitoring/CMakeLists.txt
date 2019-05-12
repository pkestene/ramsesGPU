add_library(monitoringCpu "")

target_sources (monitoringCpu
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/Timer.cpp
  )

if (PAPI_FOUND)
  target_sources (monitoringCpu
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/PapiInfo.h
    ${CMAKE_CURRENT_SOURCE_DIR}/PapiInfo.cpp
    )
  target_include_directories(monitoringCpu
    PUBLIC
    ${PAPI_INCLUDE_DIR})
  target_link_libraries(monitoringCpu
    PUBLIC
    ${PAPI_LIBRARY})
endif(PAPI_FOUND)


# alias library
add_library(RamsesGPU::monitoring ALIAS monitoringCpu)

# test executable
if (PAPI_FOUND)
  add_executable(
    PapiInfoTest
    PapiInfoTest.cpp)
  target_include_directories(PapiInfoTest 
    PUBLIC
    ${CMAKE_SOURCE_DIR}
    ${PAPI_INCLUDE_DIR})
  target_link_libraries(PapiInfoTest monitoringCpu ${PAPI_LIBRARY})
endif()
