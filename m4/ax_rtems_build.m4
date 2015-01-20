# -*- mode: autoconf -*-
#------------------------------------------------------------------------
#
# SYNOPSIS
#  AX_ENABLE_RTEMS_BSP  
#
# DESCRIPTION
#    This macro provides a configure's option --enable-rtems-bsp to specify the name
#    of the RTEMS bsp for building. This name is stored in a variable: rtems_bsp.
#
#    --enable-rtems-bsp=BSP
#      Enables the specified bsp to be built
#
AC_DEFUN([AX_ENABLE_RTEMS_BSP],
[
	AC_ARG_ENABLE(rtems-bsp,
		[AS_HELP_STRING([--enable-rtems-bsp=bsp_name],
		[Provide a RTEMS BSP name for build])],
		[case "${enableval}" in
			yes) rtems_bsp="" ;;
			no)  rtems_bsp="no" ;;
			*)   rtems_bsp="$enableval" ;;
		esac],
		[rtems_bsp=""])
])

#
# SYNOPSIS
#  AX_ENABLE_RTEMS_TOOLCHAIN
#
# DESCRIPTION
#    This macro provides a configure's option --enable-rtems-toolchain to specify
#    the name of an alternative RTEMS toolchain for building. This name is stored
#    in a variable: rtems_toolchain. The default toolchain is supposed to be the
#    official toolchain from www.rtems.org
#
#    --enable-rtems-toolchain=TOOLCHAIN
#      Enables the use of the specified toolchain
#
# Example: the sparc/Leon2 target supports the official toolchain from www.rtems.org
#          and the toolchain from www.gaisler.com
#
AC_DEFUN([AX_ENABLE_RTEMS_TOOLCHAIN],
[
	AC_ARG_ENABLE(rtems-toolchain,
		[AS_HELP_STRING([--enable-rtems-toolchain=toolchain_name],
		[Provide a RTEMS TOOLCHAIN name to use for build])],
		[case "${enableval}" in
			yes) rtems_toolchain="" ;;
			no)  rtems_toolchain="no" ;;
			*)   rtems_toolchain="$enableval" ;;
		esac],
		[rtems_toolchain="rtems.org"])
])


#
# SYNOPSIS
#   AX_RTEMS_PORTABILITY
#
# DESCRIPTION
#    This macro enables to port a module to RTEMS by seting up the environment
#    when perfoming a RTEMS cross compilation.
#    It has no effect if the host OS defined with --host is not rtems.
#
#
#    This macro calls:
#       AC_SUBST(EXEEXT)
#       AM_CONDITIONAL([BUILD_RTEMS])
#
#    This macro changes:
#       CFLAGS
#       CXXFLAGS
#       CPPFLAGS
#       LDFLAGS
#
AC_DEFUN([AX_RTEMS_PORTABILITY],
[
	AC_REQUIRE([AC_CANONICAL_HOST])
	AC_REQUIRE([AC_PROG_CC])
	AC_REQUIRE([AM_PROG_CC_C_O])

	case $host_os in
	rtems*)
		# Parse RTEMS bsp
		AX_ENABLE_RTEMS_BSP
		AC_MSG_RESULT([RTEMS BSP : ${rtems_bsp}])

		# Parse RTEMS toolchain
		AX_ENABLE_RTEMS_TOOLCHAIN
		AC_MSG_RESULT([RTEMS TOOLCHAIN : ${rtems_toolchain}])

		# For rtems.org toolchain, we use pkg-config to parse target
		# specific flags
		# AC_SUBST(PKGNAME_RTEMS_BSP,[${host_cpu}-${host_os}-${rtems_bsp}])
		# Don't forget to specify PKG_CONFIG_PATH in the environment
		# example PKG_CONFIG_PATH=/opt/rtems-4.9-powerpc/lib/pkgconfig
		if test "$rtems_toolchain" != "gaisler"; then
			PKGNAME_RTEMS_BSP=`echo ${host_cpu}-${host_os}-${rtems_bsp}`
			PKG_CHECK_MODULES([PKGNAME_RTEMS_BSP],
				[$PKGNAME_RTEMS_BSP >= 4.7],
				[AC_MSG_RESULT([pkg-config ok: [$PKGNAME_RTEMS_BSP] found])],
				[AC_MSG_FAILURE([pkg-config error: BSP [${rtems_bsp}] not found])]
			)
		fi

		# BSP specific CFLAGS
		RTEMS_BSP_CFLAGS="-D$rtems_bsp"

		# set some bsp specific flags not handled by pkg-config
		case $rtems_bsp in
		pc386)
			RTEMS_BSP_LDFLAGS=-Wl,-Ttext,0x00100000
			RTEMS_BSP_CFLAGS+=" -DQEMU"
			;;
		*)
			RTEMS_BSP_LDFLAGS=
			;;
		esac

		# sets the flags that are common to all build specifications
		CROSS_CFLAGS="-fno-builtin -DRTEMS "
		CROSS_CPPFLAGS="-DTOOL_FAMILY=gnu -DTOOL=gnu "

		# sets the flags that cpu family and toolchain specific
		if test "$rtems_toolchain" = "gaisler"; then
			# Notice : using hard floating point with gaisler toolchain
			# is the default behaviour
			case $rtems_bsp in
			leon2)
				CROSS_CFLAGS+="-qleon2 -mcpu=v8";;
			leon3)
				CROSS_CFLAGS+="-mcpu=v8";;
			erc32)
				CROSS_CFLAGS+="-tsc691";;
			*)
				AC_MSG_ERROR([unsupported RTEMS bsp: ${rtems_bsp} under gaisler toolchain. Only erc32, leon2 and leon3 are supported]) ;;
			esac
		else
			CROSS_CFLAGS+="$PKGNAME_RTEMS_BSP_CFLAGS"
		fi

		# Finaly set build FLAGS
		CFLAGS="${CFLAGS} ${CROSS_CFLAGS} ${RTEMS_BSP_CFLAGS}"
		CXXFLAGS="${CXXFLAGS} ${CROSS_CFLAGS} ${RTEMS_BSP_CFLAGS}"
		CPPFLAGS="${CPPFLAGS} ${CROSS_CPPFLAGS}"
		LDFLAGS="${LDFLAGS} ${CROSS_LDFLAGS} ${RTEMS_BSP_LDFLAGS}"
		
		EXEEXT=".exe"
		AC_SUBST([EXEEXT])
	esac

	AC_REQUIRE([AX_CROSS_BUILD])
	AM_CONDITIONAL([BUILD_RTEMS], test "$host_os" \> "rtems4.6")
])
