# ax_declare_module.m4: An m4 macro to easily create a project made of several
# modules that can be enabled or disabled.
#
# Copyright © 2013 Frederic Chateau <frederic.chateau@cea.fr>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
#
# As a special exception to the GNU General Public License, if you
# distribute this file as part of a program that contains a
# configuration script generated by Autoconf, you may include it under
# the same distribution terms that you use for the rest of that program.
#

#
# SYNOPSIS
#	AX_DECLARE_MODULE(moduleName, moduleDescription)
#
# DESCRIPTION
#	Declares a module that can be enabled or disabled using configure switches.
#   Enabled modules will install a pkgconfig file that must have the same name
#   as the module (eg: module foo-bar will require a foo-bar.pc.in file in the
#   same directory as configure.ac).
#   This macro also create a shell variable indicating whether the module has
#   been enabled or disabled and an automake conditional named after the module in
#   uppercase (eg: module foo-bar will have a USE_FOO_BAR conditional and a
#   enable_foo_bar shell variable with "yes" or "no" value).
#
#   The macro do not provide any help in managing dependencies between modules,
#   a good solution is to test that the dependencies of each enabled module
#   have not been disabled, by testing the shell variables created by each
#   module. 
#
#	Options:
#	--enable-moduleName=(yes|no)
#		Indicates whether to build the module or not.
#

AC_DEFUN([AX_DECLARE_MODULE],
[
	m4_define([moduleName], [$1])
	m4_define([varname], m4_translit(moduleName, [-], [_]))
	m4_define([switchName], [enable_]varname)
	m4_define([condName], [USE_]m4_toupper(varname))

	AC_ARG_ENABLE(moduleName,
		AS_HELP_STRING([--enable-moduleName=(yes|no)], [enables or disables $2 module.]),
		[],
		[switchName=yes]
	)
	if test "$switchName" != "yes" -a "$switchName" != "no"
	then
		AC_MSG_ERROR([invalid parameter for --enable-moduleName: $switchName])
	fi
	AC_MSG_NOTICE([moduleName enabled: $switchName])
	if test "$switchName" = "yes"
	then
		AC_CONFIG_FILES(moduleName[${DEBUG_SUFFIX}.pc:]moduleName[.pc.in])
	fi
	AM_CONDITIONAL(condName, [test "$switchName" = "yes"])
])
