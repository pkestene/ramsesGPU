#
# SYNOPSIS
#   AX_VXWORKS_BUILD
#
# DESCRIPTION
#    This macro checks that the configured module will be built for VxWorks,
#    and sets the environment accordingly.
#
# SEE
#    See AX_VXWORKS_PORTABILITY for more details about the environment
#    definition and customization.
#
AC_DEFUN([AX_VXWORKS_BUILD],
[
	if test "$host_os" != "vxworks"
	then
		AC_MSG_ERROR([this project can only be built for VxWorks. Please use --host to specify a VxWorks cross compiler.])
	fi
	AC_REQUIRE([AX_VXWORKS_PORTABILITY])
])

#
# SYNOPSIS
#   AX_VXWORKS_PORTABILITY
#
# DESCRIPTION
#    This macro enables to port a module on VxWorks by setting up the environment
#    when perfoming a VxWorks cross compilation.
#    It has no effect if the host OS defined with --host is not vxworks.
#
#    Options:
#    --enable-build-spec=BUILDSPEC
#      Enables the specified build specification instead of the default
#      one for the cross compilation host
#
#    This macro calls:
#       AC_SUBST(EXEEXT)
#       AM_CONDITIONAL([VXWORKS])
#       AM_CONDITIONAL([VXWORKS_KERNEL])
#
#    This macro changes:
#       CFLAGS
#       CXXFLAGS
#       CPPFLAGS
#       LDFLAGS
#
AC_DEFUN([AX_VXWORKS_PORTABILITY],
[
	AC_REQUIRE([AC_CANONICAL_HOST])
	AC_REQUIRE([AC_PROG_CC])
	AC_REQUIRE([AM_PROG_CC_C_O])

	if test "$host_os" = "vxworks"
	then
		# provides a way to specify the build spec
		AC_ARG_ENABLE([build-spec],
			AS_HELP_STRING([--enable-build-spec=BUILDSPEC],
				[Enables the specified build specification instead of the \
				default one for the cross compilation host]),
			[BUILDSPEC=$enableval]
		)

		# sets the flags that are shared by a cpu family and the default
		# build spec for that family
		case "$host_cpu" in
		i586)
			BUILDSPEC=${BUILDSPEC-SIMLINUXsfgnu}
			CROSS_CFLAGS="-fno-defer-pop -mtune=i486 -march=i486"
			;;
		powerpc)
			BUILDSPEC=${BUILDSPEC-PPC440sfgnu}
			CROSS_CFLAGS="-mstrict-align -mlongcall"
			;;
		sparc)
			BUILDSPEC=${BUILDSPEC-SPARCgnuv8}
			CROSS_CFLAGS=""
			;;
		*)
			AC_MSG_ERROR([unsupported CPU: ${host_cpu}. Currently i586, powerpc and sparc are supported])
			;;
		esac

		# analyses the build spec to extract its characteristics
		spec=$(echo $BUILDSPEC | sed -r 's/(@<:@A-Z0-9@:>@+)(sf)?gnu(be|le)?(v8)?(_RTP)?/\1\t\2\t\3\t\4\t\5/')
		if test "$spec" = "$BUILDSPEC"
		then
			AC_MSG_ERROR([Bad buildspec format. Valid format is <CPU>@<:@sf@:>@gnu@<:@be|le@:>@@<:@v8@:>@@<:@_RTP@:>@])
		fi

		CPU=$(echo "$spec" | cut -f 1)
		WRS_CONFIG_FP=$(echo "$spec" | cut -f 2)
		WRS_CONFIG_ENDIAN=$(echo "$spec" | cut -f 3)
		WRS_CONFIG_V8=$(echo "$spec" | cut -f 4)
		RTP=$(echo "$spec" | cut -f 5)

		# checks coherency between buildspec and host_cpu
		# TODO

		# sets the flags that are common to all build specifications
		CROSS_CFLAGS="${CROSS_CFLAGS} -fno-builtin"
		CROSS_CPPFLAGS="-DVXWORKS -DCPU=${CPU} -DTOOL_FAMILY=gnu -DTOOL=gnu${WRS_CONFIG_ENDIAN}${WRS_CONFIG_V8} -D_VSB_CONFIG_FILE=\\\"${WIND_BASE}/target/lib/h/config/vsbConfig.h\\\""

		# sets cpu specific flags
		case $CPU in
		SIMLINUX)
			;;
		SIMPENTIUM)
			CROSS_RTP_LDFLAGS="-L${WIND_USR}/lib/simpentium/SIMPENTIUM/common"
			;;
		PPC405)
			CROSS_CFLAGS="${CROSS_CFLAGS} -mcpu=405"
			;;
		PPC440)
			CROSS_CFLAGS="${CROSS_CFLAGS} -mcpu=440"
			;;
		SPARC)
			;;
		*)
			AC_MSG_ERROR([Unsupported CPU in the buildspec. Currently SIMLINUX, SIMPENTIUM_RTP, PPC440 and PPC405 are supported.])
			;;
		esac
		
		# sets floating point flags
		if test "$WRS_CONFIG_FP" = "sf"
		then
			CROSS_CFLAGS="${CROSS_CFLAGS} -msoft-float"
		else
			CROSS_CFLAGS="${CROSS_CFLAGS} -mhard-float"
		fi
		
		# sets v8 flag
		if test "$WRS_CONFIG_V8" = "v8"
		then
			CROSS_CFLAGS="${CROSS_CFLAGS} -mv8"
		fi

		# sets the flags specific to the type of build: kernel or user (rtp)
		if test -n "$RTP"
		then
			CROSS_CPPFLAGS="-I${WIND_USR}/h -I${WIND_USR}/h/wrn/coreip ${CROSS_CPPFLAGS}"
			CROSS_CFLAGS="${CROSS_CFLAGS} -mrtp"
			CROSS_LDFLAGS=$CROSS_RTP_LDFLAGS
			EXEEXT=".vxe"
		else
			CROSS_CPPFLAGS="-I${WIND_BASE}/target/h -I${WIND_BASE}/target/h/wrn/coreip -D_WRS_KERNEL ${CROSS_CPPFLAGS}"
			CROSS_CFLAGS="-nostdlib ${CROSS_CFLAGS}"
			CROSS_LDFLAGS="-r -Wl,-T${WIND_BASE}/target/h/tool/gnu/ldscripts/link.OUT"
			EXEEXT=".out"
		fi
		AC_SUBST([EXEEXT])

		CFLAGS="${CFLAGS} ${CROSS_CFLAGS}"
		CXXFLAGS="${CXXFLAGS} ${CROSS_CFLAGS}"
		CPPFLAGS="${CPPFLAGS} ${CROSS_CPPFLAGS}"
		LDFLAGS="${LDFLAGS} ${CROSS_LDFLAGS}"
	fi

	AC_REQUIRE([AX_CROSS_BUILD])

	if test -n "$BUILDSPEC"
	then
		DISTCHECK_CONFIGURE_FLAGS="$DISTCHECK_CONFIGURE_FLAGS --enable-build-spec=$BUILDSPEC"
	fi

	AM_CONDITIONAL([VXWORKS], [test "$host_os" = "vxworks"])
	AM_CONDITIONAL([VXWORKS_KERNEL], [test "$host_os" = "vxworks" -a -z "$RTP"])
])

