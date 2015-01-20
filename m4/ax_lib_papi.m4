# -*- mode: autoconf -*-
#------------------------------------------------------------------------
# AX_LIB_PAPI --
#
#	Enable/Disable PAPI - performance measuments API.
#
# Arguments:
#	none
#
# Results:
#
#	Adds the following arguments to configure:
#		--with-papi=yes|no|DIR
#
#	Defines the following variables:
#
#		PAPI_CFLAGS
#		PAPI_LDFLAGS
#------------------------------------------------------------------------

AC_DEFUN([AX_LIB_PAPI], 
[

# try to parse PAPI_ROOT location from option --with-papi
PAPI_ROOT_GIVEN=
AC_ARG_WITH(papi, 
    AS_HELP_STRING([--with-papi=@<:@=yes|no|DIR@:>@],
	[use papi, and get include files from DIR/include
		    and lib files from DIR/lib]),
	[if   test "$withval" = "no" ; then
    	    with_papi="no"
     	 elif test "$withval" = "yes"; then
     	    with_papi="yes"
     	 else
	    with_papi="yes"
     	    PAPI_ROOT_GIVEN="$withval"
     	 fi],
	 [with_papi="no"]
)

# try to check install location (either set from env variable PAPI_ROOT or
# from the argument passed to --with-papi)
if test ! "$with_papi" = "no"; then
   if test -n "$PAPI_ROOT"; then
      echo "using PAPI_ROOT from environment ($PAPI_ROOT)"
      PAPI_PATH=$PAPI_ROOT
   else 
   	if test ! "$PAPI_ROOT_GIVEN" = ""; then
	   echo "using PAPI_PATH : $PAPI_ROOT_GIVEN"
	   PAPI_PATH=$PAPI_ROOT_GIVEN
	else
	   AC_MSG_WARN( [PAPI_ROOT not found in environment] )
	   AC_MSG_WARN( [try to pass install location as arg to --with-papi] )
	   AC_MSG_WARN( [Defaulting to /usr] )
	   PAPI_PATH=/usr
        fi
   fi
   with_papi="yes"

   AC_MSG_CHECKING([for papi.h in $PAPI_PATH/include])
   if test -f $PAPI_PATH/include/papi.h; then
      AC_MSG_RESULT([yes])
   else
      AC_MSG_WARN( [papi.h not found in $PAPI_PATH/include disabling papi support ])
      with_papi=no
   fi

   THE_PAPI_LIBDIR=
   AC_MSG_CHECKING([for libpapi in $PAPI_PATH/lib])
   if test -f $PAPI_PATH/lib/libpapi.a; then
      AC_MSG_RESULT(yes)
      THE_PAPI_LIBDIR=$PAPI_PATH/lib
   elif test -f $PAPI_PATH/lib/libpapi.so; then
      AC_MSG_RESULT(yes)
      THE_PAPI_LIBDIR=$PAPI_PATH/lib
   else
      AC_MSG_WARN( [libpapi not found in $PAPI_PATH/lib disabling papi support ])
      with_papi=no
   fi

fi

# Set defaults to blank
PAPI_CFLAGS=""
PAPI_CPPFLAGS=""
PAPI_LDFLAGS=""
PAPI_INC_DIR=""

# Set pnetcdf flags
if test "x$with_papi" = "xyes"; then

      PAPI_CFLAGS="-I$PAPI_PATH/include"
      PAPI_INC_DIR="$PAPI_PATH/include"
      PAPI_CPPFLAGS="$PAPI_CFLAGS"
      PAPI_LDFLAGS="-L$THE_PAPI_LIBDIR -lpapi"
      AC_MSG_RESULT([yes])
  
fi

AC_SUBST([PAPI_PATH])
AC_SUBST([PAPI_CPPFLAGS])
AC_SUBST([PAPI_CFLAGS])
AC_SUBST([PAPI_LDFLAGS])

])
