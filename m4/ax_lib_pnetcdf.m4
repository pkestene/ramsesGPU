# -*- mode: autoconf -*-
#------------------------------------------------------------------------
# ax_lib_pnetcdf.m4: An m4 macro to detect and configure Parallel-NetCDF
#
#
# SYNOPSIS
#	AX_LIB_PNETCDF()
#
# DESCRIPTION
#	Checks the availability of Parallel-NetCDF library.
#	Options:
#	--with-pnetcdf=(path|yes|no)
#		Indicates whether to use Parallel-NetCDF or not, and the path 
#               to installation location.
#
#	This macro defines the following variables:
#		PNETCDF_VERSION
#		PNETCDF_CFLAGS
#		PNETCDF_CXXFLAGS
#		PNETCDF_LDFLAGS
#
#
#------------------------------------------------------------------------

AC_DEFUN([AX_LIB_PNETCDF],
[

AC_REQUIRE([AC_PROG_AWK])
AC_REQUIRE([AC_PROG_GREP])

# Add --with-pnetcdf configuration option.
PNETCDF_ROOT_GIVEN=
AC_ARG_WITH([pnetcdf],
    AS_HELP_STRING(
	[--with-pnetcdf@<:@=yes|no|DIR@:>@], 
	[full path to Parallel-NetCDF installation (default=no); MPI is REQUIRED to be able to use Parallel-netCDF.]),
    [if test "$withval" = "no"; then
    	with_pnetcdf="no"
     elif test "$withval" = "yes"; then
     	with_pnetcdf="yes"
     else
	with_pnetcdf="yes"
     	PNETCDF_ROOT_GIVEN="$withval"
     fi],
   [with_pnetcdf="no"]
)

# Set defaults to blank
PNETCDF_VERSION=""
PNETCDF_CFLAGS=""
PNETCDF_CPPFLAGS=""
PNETCDF_LDFLAGS=""
PNETCDF_INC_DIR=""

# try to check install location (either set from env variable PNETCDF_ROOT or
# from the argument passed to --with-pnetcdf)
if test ! "$with_pnetcdf" = "no"; then
   if test -n "$PNETCDF_ROOT"; then
      echo "using PNETCDF_ROOT from environment ($PNETCDF_ROOT)"
      PNETCDF_PATH=$PNETCDF_ROOT
   else 
   	if test ! "$PNETCDF_ROOT_GIVEN" = ""; then
	   echo "using PNETCDF_PATH : $PNETCDF_ROOT_GIVEN"
	   PNETCDF_PATH=$PNETCDF_ROOT_GIVEN
	else
		AC_MSG_WARN( [PNETCDF_ROOT not found in environment] )
		AC_MSG_WARN( [try to pass install location as arg to --with-pnetcdf] )
		AC_MSG_WARN( [Defaulting to /usr] )
	        PNETCDF_PATH=/usr
        fi
   fi
   with_pnetcdf="yes"

   AC_MSG_CHECKING([for pnetcdf.h in $PNETCDF_PATH/include])
   if test -f $PNETCDF_PATH/include/pnetcdf.h; then
      AC_MSG_RESULT([yes])
   else
      AC_MSG_WARN( [pnetcdf.h not found in $PNETCDF_PATH/include disabling pnetcdf support ])
      with_pnetcdf=no
   fi

   # AC_MSG_CHECKING([for pnetcdf.inc in $PNETCDF_PATH/include])
   # if test -f $PNETCDF_PATH/include/pnetcdf.inc
   # then AC_MSG_RESULT([yes])
   # else
   #   AC_MSG_WARN( [pnetcdf.inc not found in $PNETCDF_PATH/include \
   #                       disabling pnetcdf support ])
   #   with_pnetcdf=no
   # fi

   AC_MSG_CHECKING([for libpnetcdf.a in $PNETCDF_PATH/lib])
   if test -f $PNETCDF_PATH/lib/libpnetcdf.a; then
      AC_MSG_RESULT(yes) 
   else
      AC_MSG_WARN( [libpnetcdf.a not found in $PNETCDF_PATH/lib disabling pnetcdf support ])
      with_pnetcdf=no
   fi
fi


# Set pnetcdf flags
if test "x$with_pnetcdf" = "xyes"; then
      PNETCDF_VERSION_MAJOR=$(eval cat $PNETCDF_PATH/include/pnetcdf.h | $GREP 'define PNETCDF_VERSION_MAJOR' | $AWK '{print $[]3}')
      PNETCDF_VERSION_MINOR=$(eval cat $PNETCDF_PATH/include/pnetcdf.h | $GREP 'define PNETCDF_VERSION_MINOR' | $AWK '{print $[]3}')
      PNETCDF_VERSION_SUB=$(eval cat $PNETCDF_PATH/include/pnetcdf.h | $GREP 'define PNETCDF_VERSION_SUB' | $AWK '{print $[]3}')
      
      PNETCDF_VERSION="$PNETCDF_VERSION_MAJOR.$PNETCDF_VERSION_MINOR.$PNETCDF_VERSION_SUB"

      PNETCDF_CFLAGS="-I$PNETCDF_PATH/include"
      PNETCDF_INC_DIR="$PNETCDF_PATH/include"
      PNETCDF_CPPFLAGS="$PNETCDF_CFLAGS"
      PNETCDF_LDFLAGS="-L$PNETCDF_PATH/lib -lpnetcdf"
      AC_MSG_RESULT([yes])
  
      # MPI compiler wrapper required here !
      old_CXX=$CXX
      old_CFLAGS=$CFLAGS
      old_LDFLAGS=$LDFLAGS
      if test -z "$MPICXX"
      then
            CXX=mpicxx
      else
	    CXX=$MPICXX
      fi
      CFLAGS=$PNETCDF_CFLAGS
      LDFLAGS=$PNETCDF_LDFLAGS

      AC_CHECK_LIB(pnetcdf, ncmpi_create, [AC_DEFINE([USE_PNETCDF_LIB], [1], [Define if you want to use Parallel-NetCDF])], [
      echo ""
      echo "Error! Parallel-NetCDF is required but was not found"
      echo "       Use --with-pnetcdf=DIR to specify the Parallel-netcdf directory"
      echo "       or use environment variable PNETCDF_ROOT"
      exit -1
      ])

      CXX=$old_CXX
      CFLAGS=$old_CFLAGS
      LDFLAGS=$old_LDFLAGS
fi

AC_SUBST([PNETCDF_VERSION])
AC_SUBST([PNETCDF_CFLAGS])
AC_SUBST([PNETCDF_CPPFLAGS])
AC_SUBST([PNETCDF_LDFLAGS])

])
