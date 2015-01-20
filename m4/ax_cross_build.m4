#
# SYNOPSIS
#   AX_CROSS_BUILD
#
# DESCRIPTION
#    This macro does several common checks and configuration for cross-compiled builds.
#    It checks that when cross compiling, the compiler has the right prefix which
#    prevents configure to succeed if it cannot find the cross compiler.
#
AC_DEFUN([AX_CROSS_BUILD],
[
	if test -n "$host_alias" -a "$host_alias" != "$build_alias"
	then
		case $CXX in
		${ac_tool_prefix}*)
			;;
		*)
			AC_MSG_ERROR(Cannot find the C++ cross-compiler)
			;;
		esac
		case $CC in
		${ac_tool_prefix}*)
			;;
		*)
			AC_MSG_ERROR(Cannot find the C cross-compiler)
			;;
		esac
		
		DISTCHECK_CONFIGURE_FLAGS="$DISTCHECK_CONFIGURE_FLAGS --host=$host_alias"
		AC_SUBST([DISTCHECK_CONFIGURE_FLAGS])
	fi
	if test "$cross_compiling" = "yes"
	then
		as_test_x='test -f'
	fi
])