set (monitoring_SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/Timer.cpp
  )

if (PAPI_FOUND)
  list (APPEND monitoring_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/PapiInfo.h
    ${CMAKE_CURRENT_SOURCE_DIR}/PapiInfo.cpp
    )
endif()
    

add_library(monitoringCpu SHARED
  ${monitoring_SRCS}
  )

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
