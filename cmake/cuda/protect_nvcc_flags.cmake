# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

#[=======================================================================[.rst:

Protect flags
-------------

.. only:: html

   .. contents::

CUDA Utilities
^^^^^^^^^^^^^^

This part of the protect flags module provides a set of utilities to assist users with CUDA as a language.


It adds:


.. command:: cmake_cuda_convert_flags
    
  Take a list of flags or a target and convert the flags to pass through the CUDA compiler to 
  the host compiler by adding a LANGUAGE requirement.
  This will make the flags are only used when the language is not CUDA.

  ``PROTECT_ONLY``
    Just protect the flags, rather than passing them through to the host compiler.
  
  ``INTERFACE_TARGET <name>``
    A target to take flags from to convert

  ``LIST <name>``
    A list of flags to protect (in place).



#]=======================================================================]

# This is a private function that just converts a list
# It takes a name of a variable to modify in place
function(_CUDA_CONVERT_FLAGS flags_name)
    set(old_flags "${${flags_name}}")

    if(NOT "${old_flags}" STREQUAL "")
        # Use old flags for non-CUDA targets
        set(protected_flags "$<$<BUILD_INTERFACE:$<NOT:$<COMPILE_LANGUAGE:CUDA>>>:${old_flags}>")
        # Add -Xcompiler wrapped flags for CUDA 
        if(NOT CCF_PROTECT_ONLY)
            # These need to be comma separated now
            string(REPLACE ";" "," cuda_flags "${old_flags}")
            string(APPEND protected_flags "$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CUDA>>:-Xcompiler=${cuda_flags}>")
        endif()
        set(${flags_name} "${protected_flags}" PARENT_SCOPE)
    endif()
endfunction()


function(CMAKE_CUDA_CONVERT_FLAGS)
    cmake_parse_arguments(
        CCF
        "PROTECT_ONLY"
        ""
        "INTERFACE_TARGET;LIST"
        ${ARGN})
    
    foreach(EXISTING_TARGET IN LISTS CCF_INTERFACE_TARGET)
        get_property(old_flags TARGET ${EXISTING_TARGET} PROPERTY INTERFACE_COMPILE_OPTIONS)
        _cuda_convert_flags(old_flags "${CCF_PROTECT_ONLY}")
        set_property(TARGET ${EXISTING_TARGET} PROPERTY INTERFACE_COMPILE_OPTIONS "${old_flags}") 
    endforeach()

    foreach(EXISTING_LIST IN LISTS CCF_LIST)
        set(LOCAL_LIST "${${EXITING_LIST}}")
        _cuda_convert_flags(LOCAL_LIST "${CCF_PROTECT_ONLY}")
        set(${EXISTING_LIST} "${LOCAL_LIST}" PARENT_SCOPE)
    endforeach()
endfunction()

