# -*- mode: autoconf -*-
#------------------------------------------------------------------------
# AX_LIB_FFTW3_DOUBLE --
#
#         Find out if we have an appropriate version
#         of fftw3 (double precision)
#         Use env variable FFTW3_ROOT as a guess to installation path
#         to fftw.
#         Use env variable FFTW3_LIBDIR to set a custom lib dir (e.g.
#         /usr/lib/x86_64-linux-gnu/ for Ubuntu)	  
#
# Arguments:
#	none
#
# Results:
#
#	Adds the following arguments to configure:
#		--with-fftw3=yes|no|DIR
#		--with-fftw3-mpi=yes|no
#
#	Defines the following variables:
#
#		FFTW3_CFLAGS
#		FFTW3_CXXFLAGS
#		FFTW3_LDFLAGS
#------------------------------------------------------------------------

AC_DEFUN([AX_LIB_FFTW3_DOUBLE], 
[

# try to parse FFTW3D location from option --with-fftw3
FFTW3_ROOT_GIVEN=
AC_ARG_WITH(fftw3, 
    AS_HELP_STRING([--with-fftw3=@<:@=yes|no|DIR@:>@],
	[use fftw3, and get include files from DIR/include
		    and lib files from DIR/lib]),
	[if   test "$withval" = "no" ; then
    	    with_fftw3="no"
     	 elif test "$withval" = "yes"; then
     	    with_fftw3="yes"
     	 else
	    with_fftw3="yes"
     	    FFTW3_ROOT_GIVEN="$withval"
     	 fi],
	 [with_fftw3="no"]
)

AC_ARG_WITH(fftw3-mpi, 
    AS_HELP_STRING([--with-fftw3-mpi=@<:@=yes|no@:>@],
	[use distributed (MPI) fftw3]),
	[if   test "$withval" = "no" ; then
    	    with_fftw3_mpi="no"
     	 elif test "$withval" = "yes"; then
     	    with_fftw3_mpi="yes"
     	 else
	    with_fftw3_mpi="yes"
     	 fi],
	 [with_fftw3_mpi="no"]
)


# try to check install location (either set from env variable FFTW3_ROOT or
# from the argument passed to --with-fftw3)
if test ! "$with_fftw3" = "no"; then
   if test -n "$FFTW3_ROOT"; then
      echo "using FFTW3_ROOT from environment ($FFTW3_ROOT)"
      FFTW3_PATH=$FFTW3_ROOT
   else 
   	if test ! "$FFTW3_ROOT_GIVEN" = ""; then
	   echo "using FFTW3_PATH : $FFTW3_ROOT_GIVEN"
	   FFTW3_PATH=$FFTW3_ROOT_GIVEN
	else
	   AC_MSG_WARN( [FFTW3_ROOT not found in environment] )
	   AC_MSG_WARN( [try to pass install location as arg to --with-fftw3] )
	   AC_MSG_WARN( [Defaulting to /usr] )
	   FFTW3_PATH=/usr
        fi
   fi
   with_fftw3="yes"

   AC_MSG_CHECKING([for fftw3.h in $FFTW3_PATH/include])
   if test -f $FFTW3_PATH/include/fftw3.h; then
      AC_MSG_RESULT([yes])
   else
      AC_MSG_WARN( [fftw3.h not found in $FFTW3_PATH/include disabling fftw3 support ])
      with_fftw3=no
   fi

   THE_FFTW3_LIBDIR=
   AC_MSG_CHECKING([for libfftw3 in $FFTW3_PATH/lib])
   if test -f $FFTW3_PATH/lib/libfftw3.a; then
      AC_MSG_RESULT(yes)
      THE_FFTW3_LIBDIR=$FFTW3_PATH/lib
   elif test -f $FFTW3_PATH/lib/libfftw3.so; then
      AC_MSG_RESULT(yes)
      THE_FFTW3_LIBDIR=$FFTW3_PATH/lib
   else
      AC_MSG_WARN( [libfftw3 not found in $FFTW3_PATH/lib disabling fftw3 support ])
      with_fftw3=no
   fi

   AC_MSG_CHECKING([for libfftw3 in $FFTW3_LIBDIR])
   if test -n "$FFTW3_LIBDIR"; then
      if test -f $FFTW3_LIBDIR/libfftw3.a; then
      	 AC_MSG_RESULT(yes)
	 with_fftw3="yes"
	 THE_FFTW3_LIBDIR=$FFTW3_LIBDIR
      elif test -f $FFTW3_LIBDIR/libfftw3.so; then
         AC_MSG_RESULT(yes)
         with_fftw3="yes"
	 THE_FFTW3_LIBDIR=$FFTW3_LIBDIR
      else
	 AC_MSG_WARN( [libfftw3 not found in $FFTW3_LIBDIR disabling fftw3 support ])
      	 with_fftw3=no
      fi  
   fi

fi

# Set defaults to blank
FFTW3_CFLAGS=""
FFTW3_CPPFLAGS=""
FFTW3_LDFLAGS=""
FFTW3_INC_DIR=""

# Set pnetcdf flags
if test "x$with_fftw3" = "xyes"; then

      FFTW3_CFLAGS="-I$FFTW3_PATH/include"
      FFTW3_INC_DIR="$FFTW3_PATH/include"
      FFTW3_CPPFLAGS="$FFTW3_CFLAGS"
      FFTW3_LDFLAGS="-L$THE_FFTW3_LIBDIR -lfftw3"
      if test "x$with_fftw3_mpi" = "xyes"; then
      	 FFTW3_LDFLAGS="-L$FFTW3_PATH/lib -lfftw3_mpi -lfftw3"
      fi
      AC_MSG_RESULT([yes])
  
fi

# fftw3 is necessary to have fftw3-mpi enabled !
if test "x$with_fftw3" = "xno"; then
   with_fftw3_mpi=no
fi

AC_SUBST([FFTW3_CPPFLAGS])
AC_SUBST([FFTW3_CFLAGS])
AC_SUBST([FFTW3_LDFLAGS])
#AC_SUBST([FFTW3_MPI_LDFLAGS])

])
