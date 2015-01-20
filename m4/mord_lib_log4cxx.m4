# $Id: mord_lib_log4cxx.m4 1009 2013-01-04 15:26:07Z psizun $
#
# Copyright © 2012 sizun
#
# SYNOPSIS
#
#	MORD_LIB_LOG4CXX(ACTION-IF-FOUND, ACTION-IF-NOT-FOUND)
#
# DESCRIPTION
#
#	Test for the Apache log4cxx library.
#
#	This macro calls:
#
#	AC_SUBST(LOG4CXX_CPPFLAGS)
#	AC_SUBST(LOG4CXX_LDFLAGS)
#	AC_SUBST(LOG4CXX_LIBS)
#
#	And sets:
#
#	HAVE_LOG4CXX
#
AC_DEFUN([MORD_LIB_LOG4CXX],
[
dnl User options
	AC_ARG_WITH([log4cxx],
	[AS_HELP_STRING([--with-log4cxx@<:@=ARG@:>@],
			[use log4cxx library from a standard location (ARG=yes),
				from the specified location (ARG=<path>),
				or disable it (ARG=no) @<:@ARG=yes@:>@ ])],
		[
			if test "$withval" = "no"; then
				mord_lib_log4cxx_needed="no"
			elif test "$withval" = "yes"; then
				mord_lib_log4cxx_needed="yes"
				mord_lib_log4cxx_path=""
			else
				mord_lib_log4cxx_needed="yes"
				mord_lib_log4cxx_path="$withval"
			fi
		],
		[mord_lib_log4cxx_needed="yes"])

dnl Save variables
	saved_CPPFLAGS="$CPPFLAGS"
	saved_LDFLAGS="$LDFLAGS"
	saved_LIBS="$LIBS"
	
	CPPFLAGS=""
	LDFLAGS=""
	LIBS=""
	
	AC_REQUIRE([AC_PROG_CXX])
	AC_LANG_PUSH([C++])

	dnl Check for header
	mord_lib_log4cxx_header_found="no"
	if test "x$mord_lib_log4cxx_needed" = "xyes"; then
		if test "$mord_lib_log4cxx_path" != ""; then
			dnl Check in user provided path
			mord_lib_log4cxx_includedir=$mord_lib_log4cxx_path/include
			LOG4CXX_CPPFLAGS="-I$mord_lib_log4cxx_includedir"
			CPPFLAGS="$LOG4CXX_CPPFLAGS"
			AC_CHECK_HEADER([log4cxx/logger.h], [mord_lib_log4cxx_header_found="yes"])
		else
			dnl Check in standard locations
			mord_lib_log4cxx_includedir=""
			LOG4CXX_CPPFLAGS=""
			CPPFLAGS="$saved_CPPFLAGS $LOG4CXX_CPPFLAGS"
			AC_CHECK_HEADER([log4cxx/logger.h], [mord_lib_log4cxx_header_found="yes"])
			if test "$mord_lib_log4cxx_header_found" != "yes"; then
				dnl Check in /opt/local/include
				AS_UNSET([ac_cv_header_log4cxx_logger_h])
				mord_lib_log4cxx_includedir="/opt/local/include"
				LOG4CXX_CPPFLAGS="-I$mord_lib_log4cxx_includedir"
				CPPFLAGS="$LOG4CXX_CPPFLAGS"
				AC_CHECK_HEADER([log4cxx/logger.h], [mord_lib_log4cxx_header_found="yes"])
			fi
		fi
	fi
	
	dnl Check for library
	mord_lib_log4cxx_lib_found="no"
	if test "x$mord_lib_log4cxx_header_found" = "xyes"; then
		AC_MSG_CHECKING([for log4cxx library])
		LOG4CXX_LIBS="-llog4cxx"
		LIBS="$LOG4CXX_LIBS"
		if test "$mord_lib_log4cxx_path" != ""; then
			dnl Check in user provided path
			mord_lib_log4cxx_libdir=$mord_lib_log4cxx_path/lib
			LOG4CXX_LDFLAGS="-L$mord_lib_log4cxx_libdir"
			LDFLAGS="$LOG4CXX_LDFLAGS"
			AC_LINK_IFELSE([
				AC_LANG_PROGRAM([[@%:@include <log4cxx/logger.h>
					@%:@include <log4cxx/basicconfigurator.h>]],
					[[
					log4cxx::LoggerPtr logger = log4cxx::Logger::getLogger("mordicus");
					LOG4CXX_INFO(logger, "It works!");
					]])],
				[mord_lib_log4cxx_lib_found=yes],
				[])
		else
			dnl Check in standard locations
			mord_lib_log4cxx_includedir=""
			LOG4CXX_LDFLAGS=""
			LDFLAGS="$saved_LDFLAGS"
			AC_LINK_IFELSE([
				AC_LANG_PROGRAM([[@%:@include <log4cxx/logger.h>
					@%:@include <log4cxx/basicconfigurator.h>]],
					[[
					log4cxx::LoggerPtr logger = log4cxx::Logger::getLogger("mordicus");
					LOG4CXX_INFO(logger, "It works!");
					]])],
				[mord_lib_log4cxx_lib_found=yes],
				[])
			if test "$mord_lib_log4cxx_lib_found" != "yes"; then
				dnl Check in /opt/local/lib
				LOG4CXX_LDFLAGS="-L/opt/local/lib"
				LDFLAGS="$LOG4CXX_LDFLAGS"
				AC_LINK_IFELSE([
					AC_LANG_PROGRAM([[@%:@include <log4cxx/logger.h>
						@%:@include <log4cxx/basicconfigurator.h>]],
						[[
						log4cxx::LoggerPtr logger = log4cxx::Logger::getLogger("mordicus");
						LOG4CXX_INFO(logger, "It works!");
						]])],
					[mord_lib_log4cxx_lib_found=yes],
					[])
			fi
		fi
		AS_IF([test "$mord_lib_log4cxx_lib_found" != "no"], AC_MSG_RESULT([yes]), AC_MSG_RESULT([no]))
	fi
	
	AC_LANG_POP([C++])

dnl Restore variables
	CPPFLAGS="$saved_CPPFLAGS"
	LDFLAGS="$saved_LDFLAGS"
	LIBS="$saved_LIBS"

	if test "$mord_lib_log4cxx_lib_found" != "yes" ; then
		LOG4CXX_CPPFLAGS=""
		LOG4CXX_LDFLAGS=""
		LOG4CXX_LIBS=""
		# execute ACTION-IF-NOT-FOUND (if present):
		ifelse([$2], , :, [$2])
	else
		LOG4CXX_CPPFLAGS="-DHAVE_LOG4CXX=1 $LOG4CXX_CPPFLAGS"
		AC_SUBST([LOG4CXX_CPPFLAGS])
		AC_SUBST([LOG4CXX_LDFLAGS])
		AC_SUBST([LOG4CXX_LIBS])
		AC_DEFINE([HAVE_LOG4CXX],[1],[define if the Apache log4cxx library is available])
		# execute ACTION-IF-FOUND (if present):
		ifelse([$1], , :, [$1])
	fi
])