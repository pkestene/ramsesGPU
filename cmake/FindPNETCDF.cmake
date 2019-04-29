#
# FindPNETCDF
# -----------
#
# Find PNETCDF, a parallel I/O library for accessing NetCDF files.
#
# Possible input environment variables:
#
#   PNETCDF_ROOT - Root directory of PNETCDF.
#   PARALLEL_NETCDF_ROOT - Alias of the above.
#
# The MPI package is also found, and the include directories and linking libraries
# are appended to PNETCDF ones. By doing this, users who do not need MPI can use
# the normal compilers without linking errors.
#
# Ouput CMake variables:
#
#   PNETCDF_FOUND - True if PNETCDF was found on the system.
#   PNETCDF_INCLUDE_DIRS - List of the PNETCDF include and the dependency includes.
#   PNETCDF_LIBRARY_DIRS - List of the PNETCDF lib and the dependency libs.
#   PNETCDF_LIBRARIES - List of the PNETCDF libraries.
#
# Authors:
#
#   - Li Dong <dongli@lasg.iap.ac.cn>
#
# Source : https://github.com/dongli/geomtk
#
if (${PNETCDF_FIND_REQUIRED})
  set (required_or_not REQUIRED)
endif ()
if (${PNETCDF_FIND_QUIETLY})
  set (quiet_or_not QUIET)
endif ()

# Use pnetcdf_version command to query some PNETCDF library information.
find_program (pnetcdf_version NAMES pnetcdf_version)
if (pnetcdf_version MATCHES "NOTFOUND")
  # If pnetcdf_version can not be found, check some environment variables.
  foreach (var IN ITEMS "PNETCDF_ROOT" "PARALLEL_NETCDF_ROOT")
    if (DEFINED ENV{${var}})
      set (PNETCDF_ROOT $ENV{${var}})
      break ()
    endif ()
  endforeach ()
  if (DEFINED PNETCDF_ROOT)
    set (pnetcdf_version "${PNETCDF_ROOT}/bin/pnetcdf_version")
  endif ()
else ()
  get_filename_component (pnetcdf_bin ${pnetcdf_version} PATH)
  string (REGEX REPLACE "/bin$" "" PNETCDF_ROOT ${pnetcdf_bin})
  set (PNETCDF_FOUND TRUE)
endif ()
if (DEFINED PNETCDF_ROOT)
  list (APPEND PNETCDF_INCLUDE_DIRS "${PNETCDF_ROOT}/include")
  list (APPEND PNETCDF_LIBRARY_DIRS "${PNETCDF_ROOT}/lib")
  find_library (PNETCDF_LIBRARIES
    NAMES libpnetcdf.a
    HINTS ${PNETCDF_LIBRARY_DIRS}
    )
  # Get version string.
  execute_process (COMMAND ${pnetcdf_version} -v OUTPUT_VARIABLE output)
  string (REGEX MATCH "[0-9]+.[0-9]+.[0-9]+" PNETCDF_VERSION_STRING ${output})
  # Find the dependency MPI package and append it to PNETCDF stuffs.
  execute_process (COMMAND ${pnetcdf_version} -b OUTPUT_VARIABLE output)
  string (REGEX MATCH "MPICC: */[^ ]+" tmp ${output})
  string (REGEX REPLACE "MPICC: *" "" MPI_C_COMPILER ${tmp})
  find_package (MPI ${quiet_or_not} ${required_or_not})
  foreach (lang IN ITEMS "C" "CXX" "Fortran")
    if (${MPI_${lang}_FOUND})
      list (APPEND PNETCDF_INCLUDE_DIRS ${MPI_${lang}_INCLUDE_PATH})
      list (APPEND PNETCDF_LIBRARIES ${MPI_${lang}_LIBRARIES})
    endif ()
  endforeach ()
  list (REMOVE_DUPLICATES PNETCDF_INCLUDE_DIRS)
  list (REMOVE_DUPLICATES PNETCDF_LIBRARIES)
endif ()

include (FindPackageHandleStandardArgs)
find_package_handle_standard_args (PNETCDF FOUND_VAR PNETCDF_FOUND
  REQUIRED_VARS PNETCDF_INCLUDE_DIRS PNETCDF_LIBRARY_DIRS PNETCDF_LIBRARIES
  VERSION_VAR PNETCDF_VERSION_STRING
  )
