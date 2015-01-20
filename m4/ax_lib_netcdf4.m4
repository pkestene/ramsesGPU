# ax_lib_netcdf4.m4: An m4 macro to detect and configure NetCDF4
#
#
# SYNOPSIS
#	AX_LIB_NETCDF4()
#
# DESCRIPTION
#	Checks the availability of NetCDF4 library.
#	Options:
#	--with-netcdf=(path|yes|no)
#		Indicates whether to use NetCDF4 or not, and the path 
#               to a non-standard installation location of tool nc-config.
#
#	This macro defines the following variables:
#		NETCDF_VERSION
#		NETCDF_CFLAGS
#		NETCDF_CXXFLAGS
#		NETCDF_LDFLAGS
#
# Loosely adapted from ax_lib_hdf5 macro, with ideas from project gdl.
#
#
AC_DEFUN([AX_LIB_NETCDF4],
[

AC_REQUIRE([AC_PROG_AWK])

dnl Add a default --with-netcdf configuration option.
AC_ARG_WITH([netcdf],
    AS_HELP_STRING(
	[--with-netcdf@<:@=yes|no|DIR@:>@], 
	[full path to NetCDF nc-config tool (default=yes)]),
    [if test "$withval" = "no"; then
    	with_netcdf4="no"
     elif test "$withval" = "yes"; then
     	with_netcdf4="yes"
     else
	with_netcdf4="yes"
     	NC_CONFIG="$withval"
     fi],
   [with_netcdf4="yes"]
)

dnl Set defaults to blank
NETCDF_VERSION=""
NETCDF_CFLAGS=""
NETCDF_CPPFLAGS=""
NETCDF_LDFLAGS=""
netcdf_inc_dir=""

dnl Try to find netcdf nc-config tool and then set flags
if test "x$with_netcdf4" = "xyes"; then

   if test -z "$NC_CONFIG"; then
     AC_PATH_PROG([NC_CONFIG], nc-config, [])
   else
     AC_MSG_CHECKING([Using provided NetCDF nc-config])
     AC_MSG_RESULT([$NC_CONFIG])
   fi
   
   AC_MSG_CHECKING([for NETCDF4 libraries])
   if test ! -x "$NC_CONFIG"; then
      AC_MSG_RESULT([no])
      AC_MSG_WARN([Unable to locate serial NETCDF4 compilation helper script 'nc-config'. Please specify --with-netcdf=<LOCATION> as the full path to nc-config.
NETCDF4 support is being disabled (equivalent to --with-netcdf=no).])
      with_netcdf4="no"
      AC_MSG_RESULT([no])
   else
      dnl Look for "NetCDF Version: X.Y.Z"
      NETCDF_VERSION=$(eval $NC_CONFIG --version | $AWK '{print $[]2}')

      NETCDF_CFLAGS=$(eval $NC_CONFIG --cflags)
      netcdf_inc_dir="`$NC_CONFIG --prefix`/include"
      NETCDF_CFLAGS="$NETCDF_CFLAGS -I$netcdf_inc_dir"
      NETCDF_CPPFLAGS="$NETCDF_CFLAGS"
      NETCDF_LDFLAGS=$(eval $NC_CONFIG --libs)
      AC_MSG_RESULT([yes])
  
      AC_CHECK_LIB(netcdf, nc_open, [AC_DEFINE([USE_NETCDF4_LIB], [1], [Define if you want to use netCDF])], [
      echo ""
      echo "Error! netCDF version 3.5.1 or later is required but was not found"
      echo "       Use --with-netcdf=DIR to specify the netcdf directory tree"
      echo "       Use --with-netcdf=no  to not use it"
      echo "       Check the README or use configure --help for other libraries needed"
      echo "       (--with-xxxdir = obligatory, --with-xxx = optional (--with-xxx=no to disable))"
      echo ""
      echo "       (suitable Debian/Ubuntu package: libnetcdf-dev)"
      exit -1
      ])

      AC_CHECK_HEADERS("netcdfcpp.h", [], [
      AC_CHECK_HEADERS("$netcdf_inc_dir/netcdfcpp.h", [], [
      echo ""
      echo "Error! netCDF installation seems not to be usable"
      echo "       This suggests a conflicting netCDF-HDF4 installation, e.g."
      echo "       - uninstalling HDF4 after installation of netCDF"
      echo "       - installing netCDF before HDF4" 
      exit -1
        ])
      ])
   fi

fi

AC_SUBST([NETCDF_VERSION])
AC_SUBST([NETCDF_CFLAGS])
AC_SUBST([NETCDF_CPPFLAGS])
AC_SUBST([NETCDF_LDFLAGS])

])
