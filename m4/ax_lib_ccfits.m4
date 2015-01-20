# ax_lib_ccfits.m4: An M4 macro to detect the CCfits library
# See http://heasarc.gsfc.nasa.gov/fitsio/CCfits/
# $Id: ax_lib_ccfits.m4 1406 2013-11-18 07:45:32Z psizun $
#
# SYNOPSIS
#	AX_LIB_CCFITS([ACTION-IF-FOUND], [ACTION-IF-NOT-FOUND])
#
# DESCRIPTION
#	Checks the existence of the CCfits library.
#
#   If CCfits is successfully found, this macro calls
#
#     AC_SUBST(CCFITS_CPPFLAGS)
#     AC_SUBST(CCFITS_LDFLAGS)
#     AC_SUBST(CCFITS_LIBS)
#     AC_DEFINE(HAVE_CCFITS)
#
AC_DEFUN([AX_LIB_CCFITS],
[
dnl Options
	AC_ARG_WITH([ccfits-includedir],
  		[AS_HELP_STRING([--with-ccfits-includedir=DIR], [include directory of the CCfits library])],
  		[ccfits_includedir="$withval"]
	)
	AC_ARG_WITH([ccfits-libdir],
  		[AS_HELP_STRING([--with-ccfits-libdir=DIR], [location of the CCfits library])],
  		[ccfits_libdir="$withval"]
	)

dnl Save initial flags
	saved_CPPFLAGS="$CPPFLAGS"
	saved_CXXFLAGS="$CXXFLAGS"
	saved_LDFLAGS="$LDFLAGS"
	saved_LIBS="$LIBS"

dnl Set flags	
	AS_IF([test "x$ccfits_includedir" != "x" ], [CCFITS_CPPFLAGS="-I$ccfits_includedir"])
	AS_IF([test "x$ccfits_libdir" != "x" ], [CCFITS_LDFLAGS="-L$ccfits_libdir"])
	CCFITS_LIBS="-lCCfits"
	CPPFLAGS="$CPPFLAGS $CCFITS_CPPFLAGS"
	LDFLAGS="$LDFLAGS $CCFITS_LDFLAGS"
	LIBS="$LIBS $CCFITS_LIBS"

	AC_REQUIRE([AC_PROG_CXX])
	AC_LANG_PUSH([C++])
	have_ccfits_lib=yes

	AC_CHECK_LIB(CCfits,main,[],[have_ccfits_lib=no])
	AC_CHECK_HEADERS([CCfits/CCfits],[],[have_ccfits_lib=no])

	AC_LANG_POP([C++])

dnl Restore initial flags
	CPPFLAGS="$saved_CPPFLAGS"
	CXXFLAGS="$saved_CXXFLAGS"
	LDFLAGS="$saved_LDFLAGS"
	LIBS="$saved_LIBS"

	AM_CONDITIONAL([HAVE_CCFITS], [test x$have_ccfits_lib = xyes])
	if test "$have_ccfits_lib" != "yes" ; then
		CCFITS_CPPFLAGS=""
		CCFITS_LDFLAGS=""
		CCFITS_LIBS=""
		# execute ACTION-IF-NOT-FOUND (if present):
		ifelse([$2], , :, [$2])
	else
		CCFITS_CPPFLAGS="-DHAVE_CCFITS=1 $CCFITS_CPPFLAGS"
		AC_SUBST([CCFITS_CPPFLAGS])
		AC_SUBST([CCFITS_LDFLAGS])
		AC_SUBST([CCFITS_LIBS])
		AC_DEFINE([HAVE_CCFITS],[1],[defined if the CCfits library is available])
		# execute ACTION-IF-FOUND (if present):
		ifelse([$1], , :, [$1])
	fi
])
