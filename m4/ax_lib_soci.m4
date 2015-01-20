#
# SYNOPSIS
#   AX_LIB_SOCI(RELEASE)
#
# DESCRIPTION
#    Checks the existence of SOCI libraries of a particular version.
#    RELEASE
#       release suffix for libraries. eg: 2_0, 2_2, etc.
#
#    Options:
#    --with-soci=path
#       Indicates the directory where SOCI is installed. The directory path/lib
#       must contain SOCI libraries, and the directory path/include the headers.
#
#    This macro calls:
#       AC_SUBST(SOCI_CPPFLAGS)
#       AC_SUBST(SOCI_LDFLAGS)
#       AC_DEFINE(HAVE_FIREBIRD)
#       AC_DEFINE(HAVE_MYSQL)
#       AC_DEFINE(HAVE_ODBC)
#       AC_DEFINE(HAVE_ORACLE)
#       AC_DEFINE(HAVE_POSTGRESQL)
#       AC_DEFINE(HAVE_SQLITE3)
#
#

AC_DEFUN([AX_LIB_SOCI],
[
	if test -z "$1" -o -z "%2"
	then
		AC_MSG_ERROR([The release suffix (eg: 2_2) and the build suffix (eg: gcc-g) must be specified in parameter])
	else
		soci_release="$1"
		soci_cc_suffix="$2"
		soci_core_lib="soci_core-${soci_cc_suffix}-${soci_release}"
	fi
	AC_MSG_CHECKING(for SOCI ${soci_release})
	AC_ARG_WITH([soci],
		AC_HELP_STRING([--with-soci@<:@=DIR@:>@],
			[Specifies where the SOCI library is installed.
			If undefined, searches in /opt /usr /usr/local]),
		[
			if test "${withval}" = "no" || test "${withval}" = "yes"; then
				AC_MSG_ERROR([The path to SOCI is needed, but disabling it is impossible])
			else
				soci_dir="${withval}"
			fi
		],
		[soci_dir=""]
	)

	# if the directory is not specified, search in a standard location
	if test -z ${soci_dir}
	then
		for soci_dir_tmp in /usr /usr/local /opt
		do
			if test -d "${soci_dir_tmp}/include/soci" && test -r "${soci_dir_tmp}/include/soci"
			then
				soci_dir=${soci_dir_tmp}
				break;
			fi
		done
	fi
	# check if we still don't know where to find SOCI
	if test -z "${soci_dir}"
	then
		AC_MSG_ERROR([SOCI directory is undefined and cannot be found in a standard location.])
	else
		soci_incl_dir=${soci_dir}/include
		soci_lib_dir=${soci_dir}/lib
	fi
	# Check if the core library exists
	if ! test -r "${soci_lib_dir}/lib${soci_core_lib}.la"
	then
		AC_MSG_ERROR([Cannot find SOCI Core library for release ${RELEASE}.])
	fi
	AC_MSG_RESULT(yes)
	AC_DEFINE([HAVE_SOCI], [1], [defined if the SOCI library is available])
	SOCI_CPPFLAGS="-I${soci_incl_dir}/soci"
	SOCI_LDFLAGS="-L${soci_lib_dir} -l${soci_core_lib}"

	# Look for compiled backends
	for backend in firebird mysql odbc oracle postgresql sqlite3
	do
		AC_MSG_CHECKING([for ${backend} backend])

		# creates a variable name like "HAVE_MYSQL"
		backend_var=`echo "have_${backend}" | tr '[[:lower:]]' '[[:upper:]]'`

		# checks if a libtool library for the backend exists
		if test -r "${soci_lib_dir}/libsoci_${backend}-${soci_cc_suffix}-${soci_release}.la"
		then
			# creates a variable whose name is the string in $backend_var and assigns it.
			export ${backend_var}="yes"
			SOCI_LDFLAGS="${SOCI_LDFLAGS} -lsoci_${backend}-${soci_cc_suffix}-${soci_release}"
		else
			export ${backend_var}="no"
		fi

		AC_MSG_RESULT([${!backend_var}])

		if test "${!backend_var}" = "yes"
		then
			AC_DEFINE_UNQUOTED([${backend_var}])
		fi
	done

	# This test always fails but it declares possible backend variables to autoheader.
	# Otherwise they do not appear in the config.h.in and are discarded from the config.h
	if test -n "${dummy}"
	then
		AC_DEFINE([HAVE_FIREBIRD], [], [defined if the firebird SOCI backend is available])
		AC_DEFINE([HAVE_MYSQL], [], [defined if the MySql SOCI backend is available])
		AC_DEFINE([HAVE_ODBC], [], [defined if the ODBC SOCI backend is available])
		AC_DEFINE([HAVE_ORACLE], [], [defined if the Oracle SOCI backend is available])
		AC_DEFINE([HAVE_POSTGRESQL], [], [defined if the PostgreSql SOCI backend is available])
		AC_DEFINE([HAVE_SQLITE3], [], [defined if the SQLite3 SOCI backend is available])
	fi

	AC_SUBST(SOCI_CPPFLAGS)
	AC_SUBST(SOCI_LDFLAGS)
])
