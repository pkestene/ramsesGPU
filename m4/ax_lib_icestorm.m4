# $Id$
#
# Copyright © 2012 sizun
#
# SYNOPSIS
#
#	AX_LIB_ICESTORM(ACTION-IF-FOUND, ACTION-IF-NOT-FOUND)
#
# DESCRIPTION
#
#	Test for IceStorm C++ library.
#	The macro requires a preceding call to AX_LIB_ICE_OR_ICEE.
#
#	This macro calls:
#
#	AC_SUBST(ICESTORM_CPPFLAGS)
#	AC_SUBST(ICESTORM_LIBS)
#
#	And sets:
#
#	HAVE_ICESTORM
#
AC_DEFUN([AX_LIB_ICESTORM],
[
	AC_MSG_CHECKING([for IceStorm library])

	saved_CPPFLAGS="$CPPFLAGS"
	saved_CXXFLAGS="$CXXFLAGS"
	saved_LDFLAGS="$LDFLAGS"
	saved_LIBS="$LIBS"
	
	CPPFLAGS="$CPPFLAGS $ICE_CPPFLAGS"
	CXXFLAGS="$CXXFLAGS $PTHREAD_CFLAGS"
	LDFLAGS="$ICE_LDFLAGS"
	ICESTORM_LIBS="-lIceStorm"
	LIBS="$ICE_LIBS $PTHREAD_LIBS $ICESTORM_LIBS"

	AC_REQUIRE([AC_PROG_CXX])
	AC_LANG_PUSH([C++])
	have_icestorm_lib=no
	AC_LINK_IFELSE([
		AC_LANG_PROGRAM([[@%:@include <$ICE_EDITION/$ICE_EDITION.h>
		@%:@include <IceStorm/IceStorm.h>]],
		[[
			int argc = 0;
			char** argv=0;
			Ice::CommunicatorPtr ic = Ice::initialize(argc, argv);
			Ice::ObjectPrx obj = ic->stringToProxy("IceStorm/TopicManager:tcp -p 9999");
			IceStorm::TopicManagerPrx topicManager = IceStorm::TopicManagerPrx::checkedCast(obj);
			IceStorm::TopicPrx topic = 0;
			try
			{
				topic = topicManager->retrieve("Weather");
			}
			catch (const IceStorm::NoSuchTopic&)
			{
			}]])],
		[have_icestorm_lib=yes],
		[])
dnl Check for MacPorts name of IceStorm library (after revision 110485 of port zeroc-ice34)
	if test "$have_icestorm_lib" != "yes" ; then
		ICESTORM_LIBS="-lZerocIceStorm"
		LIBS="$ICE_LIBS $PTHREAD_LIBS $ICESTORM_LIBS"
		AC_LINK_IFELSE([
			AC_LANG_PROGRAM([[@%:@include <$ICE_EDITION/$ICE_EDITION.h>
			@%:@include <IceStorm/IceStorm.h>]],
			[[
				int argc = 0;
				char** argv=0;
				Ice::CommunicatorPtr ic = Ice::initialize(argc, argv);
				Ice::ObjectPrx obj = ic->stringToProxy("IceStorm/TopicManager:tcp -p 9999");
				IceStorm::TopicManagerPrx topicManager = IceStorm::TopicManagerPrx::checkedCast(obj);
				IceStorm::TopicPrx topic = 0;
				try
				{
					topic = topicManager->retrieve("Weather");
				}
				catch (const IceStorm::NoSuchTopic&)
				{
				}]])],
		[have_icestorm_lib=yes],
		[])
	fi
	AC_LANG_POP([C++])
	
	CPPFLAGS="$saved_CPPFLAGS"
	CXXFLAGS="$saved_CXXFLAGS"
	LDFLAGS="$saved_LDFLAGS"
	LIBS="$saved_LIBS"
	
	if test "$have_icestorm_lib" != "yes" ; then
		AC_MSG_RESULT([no])
		ICESTORM_CPPFLAGS=""
		ICESTORM_LIBS=""
		# execute ACTION-IF-NOT-FOUND (if present):
		ifelse([$2], , :, [$2])
	else
		ICESTORM_CPPFLAGS="-DHAVE_ICESTORM=1"
		AC_MSG_RESULT([yes])
		AC_SUBST([ICESTORM_CPPFLAGS])
		AC_SUBST([ICESTORM_LIBS])
		AC_DEFINE([HAVE_ICESTORM],[1],[define if the IceStorm library is available])
		# execute ACTION-IF-FOUND (if present):
		ifelse([$1], , :, [$1])
	fi
])
