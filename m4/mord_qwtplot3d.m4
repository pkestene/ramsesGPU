# $Id$
#
# Copyright © 2012 sizun
#
# SYNOPSIS
#
#	MORD_QWT_PLOT_3D(ACTION-IF-FOUND, ACTION-IF-NOT-FOUND)
#
# DESCRIPTION
#
#	Test for QwtPlot3d C++ library.
#	The macro requires a preceding call to AT_WITH_QT.
#
#	This macro calls:
#
#	AC_SUBST(QWT_PLOT_3D_CPPFLAGS)
#	AC_SUBST(QWT_PLOT_3D_LDFLAGS)
#	AC_SUBST(QWT_PLOT_3D_LIBS)
#
#	And sets:
#
#	HAVE_QWT_PLOT_3D
#

AC_DEFUN([MORD_QWT_PLOT_3D],
[
	AC_ARG_WITH([qwt-plot-3d],
  		[AS_HELP_STRING([--with-qwt-plot-3d@<:@=ARG@:>@],
			[use QwtPlot3d library from a standard location (ARG=yes),
	 			from the specified location (ARG=<path>),
				or disable it (ARG=no) @<:@ARG=yes@:>@ ])],
		[
			if test "$withval" = "no"; then
				want_qwt_plot_3d="no"
			elif test "$withval" = "yes"; then
				want_qwt_plot_3d="yes"
				ac_qwt_plot_3d_path=""
			else
				want_qwt_plot_3d="yes"
				ac_qwt_plot_3d_path="$withval"
			fi
		],
		[want_qwt_plot_3d="yes"])
	
	succeeded=no
	if test "x$want_qwt_plot_3d" = "xyes"; then
		dnl Define library subdirectories
		libsubdirs="lib"
		if test `uname -m` = x86_64; then
			libsubdirs="lib64 lib lib64"
		fi
		libsubdir="lib"
		
		dnl Define include subdirectories
		incdirnames="qwtplot3d qwtplot3d-qt4"
		
		dnl first we check the system location for QwtPlot3d library
		AC_MSG_CHECKING([for QwtPlot3d])
		qwt_plot_3d_path_found="no"
		if test "$ac_qwt_plot_3d_path" != ""; then
			qwt_plot_3d_libdir=$ac_qwt_plot_3d_path/$libsubdir
			for incdirname in $incdirnames ; do
				if ls "$ac_qwt_plot_3d_path/include/$incdirname/qwt3d_"* >/dev/null 2>&1 ; then break; fi
			done
			qwt_plot_3d_incdir=$ac_qwt_plot_3d_path/include/$incdirname
			qwt_plot_3d_path_found="yes"
		elif test "$cross_compiling" != yes; then
			for ac_qwt_plot_3d_path_tmp in /usr /usr/local /opt /opt/local ; do
				for incdirname in $incdirnames ; do
					if test -d "$ac_qwt_plot_3d_path_tmp/include/$incdirname" && test -r "$ac_qwt_plot_3d_path_tmp/include/$incdirname"; then
						qwt_plot_3d_incdir="$ac_qwt_plot_3d_path_tmp/include/$incdirname"
						break;
					fi
				done
				for libsubdir in $libsubdirs ; do
					if ls "$ac_qwt_plot_3d_path_tmp/$libsubdir/libqwtplot3d"* >/dev/null 2>&1 ; then
						qwt_plot_3d_libdir=$ac_qwt_plot_3d_path_tmp/$libsubdir
						qwt_plot_3d_path_found="yes"
						break;
					fi
				done
				if test "$qwt_plot_3d_path_found" != "no"; then break; fi
			done
		fi
		AS_IF([test "$qwt_plot_3d_path_found" != "no"], AC_MSG_RESULT([yes]), AC_MSG_RESULT([no]))

		qwt_plot_3d_header_found="no"
		if test "$qwt_plot_3d_path_found" != "no"; then
			AC_MSG_CHECKING([for QwtPlot3d headers])
			if test -f "$qwt_plot_3d_incdir/qwt3d_plot.h"; then
				AC_MSG_RESULT([$qwt_plot_3d_incdir])
				qwt_plot_3d_header_found="yes"
			else
				AC_MSG_RESULT([no])
			fi
		fi 
		
		qwt_plot_3d_lib_found="no"
		if test "$qwt_plot_3d_header_found" != "no"; then
			AC_MSG_CHECKING([for QwtPlot3d library])

			QWT_PLOT_3D_LDFLAGS="-L$qwt_plot_3d_libdir"
			QWT_PLOT_3D_CPPFLAGS="-DHAVE_QWT_PLOT_3D -I$qwt_plot_3d_incdir"
		
			CPPFLAGS_SAVED="$CPPFLAGS"
			CPPFLAGS="$QT_CPPFLAGS $QWT_PLOT_3D_CPPFLAGS $CPPFLAGS"
			export CPPFLAGS
			
			CXXFLAGS_SAVED="$CXXFLAGS"
			CXXFLAGS=""
			export CXXFLAGS

			LDFLAGS_SAVED="$LDFLAGS"
			LDFLAGS="$QT_LDFLAGS $QWT_PLOT_3D_LDFLAGS $LDFLAGS"
			export LDFLAGS
			
			LIBS_SAVED="$LIBS"
			
			dnl Define library names
			libnames="qwtplot3d qwtplot3d-qt4"

			for libname in $libnames ; do
				QWT_PLOT_3D_LIBS="-l$libname -lQtOpenGL -lGL -lGLU"
				LIBS="$QT_LIBS $QWT_PLOT_3D_LIBS"
				
				AC_REQUIRE([AC_PROG_CXX])
				AC_LANG_PUSH([C++])
				AC_LINK_IFELSE([
					AC_LANG_PROGRAM([[@%:@include <QApplication>
						@%:@include <qwt3d_surfaceplot.h>]],
						[[
						int argc = 0;
						char** argv=0;
						QApplication a(argc, argv);
						Qwt3D::SurfacePlot plot;
						plot.resize(800,600);
						plot.show();
						return a.exec();]])],
					[qwt_plot_3d_lib_found=yes
					 qwt_plot_3d_lib=$libname],
					[])
				AC_LANG_POP([C++])
				if test "$qwt_plot_3d_lib_found" != "no"; then break; fi
			done
			AS_IF([test "$qwt_plot_3d_lib_found" != "no"], AC_MSG_RESULT([-L$qwt_plot_3d_libdir -l$qwt_plot_3d_lib]), AC_MSG_RESULT([no]))
		fi
	fi
	
	CPPFLAGS="$CPPFLAGS_SAVED"
	CXXFLAGS="$CXXFLAGS_SAVED"
	LDFLAGS="$LDFLAGS_SAVED"
	LIBS="$LIBS_SAVED"
	
	if test "$qwt_plot_3d_lib_found" != "yes" ; then
		QWT_PLOT_3D_CPPFLAGS=""
		QWT_PLOT_3D_LDFLAGS=""
		QWT_PLOT_3D_LIBS=""
		# execute ACTION-IF-NOT-FOUND (if present):
		ifelse([$2], , :, [$2])
	else
		AC_SUBST([QWT_PLOT_3D_CPPFLAGS])
		AC_SUBST([QWT_PLOT_3D_LDFLAGS])
		AC_SUBST([QWT_PLOT_3D_LIBS])
		AC_DEFINE([HAVE_QWT_PLOT_3D],[1],[define if the QwtPlot3d library is available])
		# execute ACTION-IF-FOUND (if present):
		ifelse([$1], , :, [$1])
	fi
])