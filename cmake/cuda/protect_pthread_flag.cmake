#
# flags '-pthreads' is not support by nvcc, replace with
#  '-Xcompiler -pthread'
#

function(CUDA_PROTECT_PTHREAD_FLAG EXISTING_TARGET)

  get_property(olds_flags TARGET ${EXISTING_TARGET} PROPERTY INTERFACE_INCLUDE_DIRECTORIES)
  if(NOT "${old_flags}" STREQUAL "")
    string(REPLACE "-pthread" "-Xcompiler -pthread" new_flags "${old_flags}")
    set_property(TARGET ${EXISTING_TARGET} PROPERTY INTERFACE_INCLUDE_DIRECTORIES
      "$<$<BUILD_INTERFACE:$<NOT:$<COMPILE_LANGUAGE:CUDA>>>:${old_flags}>$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CUDA>>:-Xcompiler=${new_flags}>"
      )
  endif()

  get_property(olds_flags TARGET ${EXISTING_TARGET} PROPERTY INTERFACE_COMPILE_OPTIONS)
  if(NOT "${old_flags}" STREQUAL "")
    string(REPLACE "-pthread" "-Xcompiler -pthread" new_flags "${old_flags}")
    set_property(TARGET ${EXISTING_TARGET} PROPERTY INTERFACE_COMPILE_OPTIONS
      "$<$<BUILD_INTERFACE:$<NOT:$<COMPILE_LANGUAGE:CUDA>>>:${old_flags}>$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CUDA>>:-Xcompiler=${new_flags}>"
      )
  endif()
  
  # debug
  get_property(current_flags TARGET ${EXISTING_TARGET} PROPERTY INTERFACE_INCLUDE_DIRECTORIES)
  message("DEBUG : TARGET=${EXISTING_TARGET} compile flags=${current_flags}")
  get_property(current_flags TARGET ${EXISTING_TARGET} PROPERTY INTERFACE_COMPILE_OPTIONS)
  message("DEBUG : TARGET=${EXISTING_TARGET} compile flags=${current_flags}")

endfunction()

