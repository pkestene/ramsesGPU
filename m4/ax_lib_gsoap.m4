# ax_lib_gsoap.m4: An M4 macro to detect the gSOAP toolkit
#
# $Id: ax_lib_gsoap.m4 1526 2014-02-21 15:52:19Z psizun $
#
# SYNOPSIS
#	AX_LIB_GSOAP([ACTION-IF-FOUND], [ACTION-IF-NOT-FOUND])
#
# DESCRIPTION
#	Checks the existence of the gSOAP toolkit.
#
#   If gSOAP is successfully found, this macro calls
#
#     AC_SUBST(GSOAP_CFLAGS)
#     AC_SUBST(GSOAP_LIBS)
# 	  AC_SUBST(GSOAP_VERSION)
#     AC_DEFINE(HAVE_GSOAP)
#
# serial 2
#
AC_DEFUN([AX_LIB_GSOAP],
[
dnl Check for gSOAP compiler
	AC_ARG_WITH([soapcpp2-bin],
		AS_HELP_STRING([--with-soapcpp2-bin=DIR], [location of gSOAP compiler soapcpp2]),
		[SOAP_CPP2_DIR="$withval"])
	AC_ARG_VAR([SOAP_CPP2], [Path of gSOAP compiler soapcpp2])
	AC_PATH_PROG([SOAP_CPP2], [soapcpp2], [no], [$PATH$PATH_SEPARATOR$SOAP_CPP2_DIR])

dnl Check for gSOAP library
	PKG_CHECK_MODULES([GSOAP], gsoap++, [have_gsoap_lib=yes], [have_gsoap_lib=no])

dnl Check version
	if test x"$have_gsoap_lib" = "xyes"
	then
		gsoap_include_dir=`$PKG_CONFIG --variable=includedir gsoap++`
		AC_CHECK_FILE([${gsoap_include_dir}/stdsoap2.h],
		[
			AC_PROG_GREP
			AC_PROG_AWK
			AC_MSG_CHECKING([for gSOAP version])
			gsoap_version=`$GREP 'define GSOAP_H_VERSION' ${gsoap_include_dir}/stdsoap2.h | $AWK '{ print $NF; }'`
			AS_IF([test -z "$gsoap_version"],[gsoap_version=`$GREP 'define GSOAP_VERSION' ${gsoap_include_dir}/stdsoap2.h | $AWK '{ print $NF; }'`])
			AS_IF([test -z "$gsoap_version"],
				[AC_MSG_RESULT([not found])
				 gsoap_version=0],
				[AC_MSG_RESULT([$gsoap_version])])
			AC_SUBST([GSOAP_VERSION], [$gsoap_version])
		],
		[AC_MSG_WARN([Could not find version of the gSOAP toolkit])])
	fi

	if test x"$SOAP_CPP2" != "xno" -a x"$have_gsoap_lib" = "xyes"
	then
		AC_SUBST([GSOAP_CFLAGS])
		AC_SUBST([GSOAP_LIBS])
		AC_DEFINE([HAVE_GSOAP],[1],[defined if the gSOAP toolkit is available])
		dnl execute ACTION-IF-FOUND
		ifelse([$1], , :, [$1])
	else
		dnl execute ACTION-IF-NOT-FOUND
		ifelse([$2], , :, [$2])
	fi
])
