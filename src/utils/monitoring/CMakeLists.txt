add_library(monitoringCpu "")

target_sources (monitoringCpu
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/Timer.cpp
  )

if (RAMSESGPU_PAPI_FOUND)
  target_sources (monitoringCpu
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/PapiInfo.h
    ${CMAKE_CURRENT_SOURCE_DIR}/PapiInfo.cpp
    )
  target_link_libraries(monitoringCpu
    PUBLIC
    RamsesGPU::papi)
endif(RAMSESGPU_PAPI_FOUND)

# alias library
add_library(RamsesGPU::monitoring ALIAS monitoringCpu)

# test executable
if (RAMSESGPU_PAPI_FOUND)
  add_executable(
    PapiInfoTest
    PapiInfoTest.cpp)
  target_link_libraries(PapiInfoTest RamsesGPU::monitoring)
endif()
