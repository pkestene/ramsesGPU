#
# SYNOPSIS
#   AX_WINDOWS_BUILD
#
# DESCRIPTION
#    This macro checks that the configured module will be built for Windows,
#    and sets the environment accordingly.
#
# SEE
#    See AX_WINDOWS_PORTABILITY for more details about the environment
#    definition and customization.
#
AC_DEFUN([AX_WINDOWS_BUILD],
[
	case $host_os in
    *mingw32*) 
		;;
    *) AC_MSG_ERROR([this project can only be built for Windows (*mingw32*). Please use --host to specify a Windows cross compiler.])
		;;
    esac
	AC_REQUIRE([AX_WINDOWS_PORTABILITY])
])

#
# SYNOPSIS
#   AX_WINDOWS_PORTABILITY
#
# DESCRIPTION
#    This macro enables to port a module on Windows by setting up the environment
#    when perfoming a Windows cross compilation.
#    It has no effect if the host OS defined with --host is not mingw32msvc.
#
#    This macro calls:
#       AM_CONDITIONAL([WINDOWS])
#
#    This macro changes:
#       CPPFLAGS
#       LDFLAGS
#
AC_DEFUN([AX_WINDOWS_PORTABILITY],
[
	AC_REQUIRE([AC_CANONICAL_HOST])
	AC_REQUIRE([AC_PROG_CC])
	AC_REQUIRE([AM_PROG_CC_C_O])

	case $host_os in
    *mingw32*)
    	use_windows="yes"
		# sets the flags necessary for windows cross-compilation
		CPPFLAGS="${CPPFLAGS} -U__STRICT_ANSI__"
		LDFLAGS="${LDFLAGS} -Wl,--enable-auto-import"
		;;
    *)
    	use_windows="no"
    	;;
    esac

	AC_REQUIRE([AX_CROSS_BUILD])
	AM_CONDITIONAL([WINDOWS], [test "$use_windows" = "yes"])
])

# Note: I have made some attempts producing a shared DLL with mingw32
# catching exceptions at the DLL-exe boundary
# without success, but I arrived with a working program, without exception crossing using the following options:
# CXXFLAGS="-fno-strict-aliasing -mthreads -shared-libgcc -D_DLL"
# LDFLAGS="-Wl,--enable-auto-import -shared -shared-libgcc"
# LIBS="-lws2_32 -lrpcrt4 -lgcc_s  -L/usr/lib/gcc/i586-mingw32msvc/4.2.1-sjlj  -lstdc++_s 
#
# I have also replaced the post-dependency CXX flags at the end of configure.ac with:
# postdeps_CXX="-lmingw32 -lmoldname -lmingwex -lmsvcrt -luser32 -lkernel32 -ladvapi32 -lshell32 -lmingw32 -lmoldname -lmingwex -lmsvcrt"

