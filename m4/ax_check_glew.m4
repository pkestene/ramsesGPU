# ax_check_glew.m4 : m4 macro to detect GLEW (mostly adapted from ax_check_glut.m4)
#
# SYNOPSIS
#
#   AX_CHECK_GLEW
#
# DESCRIPTION
#
#   Check for the OpenGL Extension Wrangler Library (GLEW). 
#   If GLEW is found, the required compiler and linker flags
#   are included in the output variables "GLEW_CFLAGS" and "GLEW_LIBS",
#   respectively. If GLEW is not found, "no_glew" is set to "yes".
#
#   If the header "GL/glew.h" is found, "HAVE_GL_GLEW_H" is defined.
#
# LICENSE
#
#   Copyright (c) 2010 Pierre Kestener <pierre.kestener@cea.fr>
#
#   This program is free software; you can redistribute it and/or modify it
#   under the terms of the GNU General Public License as published by the
#   Free Software Foundation; either version 2 of the License, or (at your
#   option) any later version.
#
#   This program is distributed in the hope that it will be useful, but
#   WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
#   Public License for more details.
#
#   You should have received a copy of the GNU General Public License along
#   with this program. If not, see <http://www.gnu.org/licenses/>.
#
#   As a special exception, the respective Autoconf Macro's copyright owner
#   gives unlimited permission to copy, distribute and modify the configure
#   scripts that are the output of Autoconf when processing the Macro. You
#   need not follow the terms of the GNU General Public License when using
#   or distributing such scripts, even though portions of the text of the
#   Macro appear in them. The GNU General Public License (GPL) does govern
#   all other use of the material that constitutes the Autoconf Macro.
#
#   This special exception to the GPL applies to versions of the Autoconf
#   Macro released by the Autoconf Archive. When you make and distribute a
#   modified version of the Autoconf Macro, you may extend this special
#   exception to the GPL to apply to your modified version as well.

AC_DEFUN([AX_CHECK_GLEW],
[AC_REQUIRE([AX_CHECK_GLU])dnl
AC_REQUIRE([AC_PATH_XTRA])dnl

ax_save_CPPFLAGS="${CPPFLAGS}"
CPPFLAGS="${GLU_CFLAGS} ${CPPFLAGS}"
AC_CHECK_HEADERS([GL/glew.h])
CPPFLAGS="${ax_save_CPPFLAGS}"

GLEW_CFLAGS=${GLU_CFLAGS}
GLEW_LIBS=${GLU_LIBS}

m4_define([AX_CHECK_GLEW_PROGRAM],
          [AC_LANG_PROGRAM([[
# if HAVE_WINDOWS_H && defined(_WIN32)
#   include <windows.h>
# endif
# ifdef HAVE_GL_GLEW_H
#   include <GL/glew.h>
# else
#   error no glew.h
# endif]],
                           [[glewInit()]])])

#
# If X is present, assume GLEW depends on it.
#
AS_IF([test X$no_x != Xyes],
      [GLEW_LIBS="${X_PRE_LIBS} -lXmu -lXi ${X_EXTRA_LIBS} ${GLEW_LIBS}"])

AC_CACHE_CHECK([for GLEW library], [ax_cv_check_glew_libglew],
[ax_cv_check_glew_libglew="no"
AC_LANG_PUSH(C)
ax_save_CPPFLAGS="${CPPFLAGS}"
CPPFLAGS="${GLEW_CFLAGS} ${CPPFLAGS}"
ax_save_LIBS="${LIBS}"
LIBS=""
ax_check_libs="-lGLEW"
for ax_lib in ${ax_check_libs}; do
  AS_IF([test X$ax_compiler_ms = Xyes],
        [ax_try_lib=`echo $ax_lib | sed -e 's/^-l//' -e 's/$/.lib/'`],
        [ax_try_lib="${ax_lib}"])
  LIBS="${ax_try_lib} ${GLEW_LIBS} ${ax_save_LIBS}"
  AC_LINK_IFELSE([AX_CHECK_GLEW_PROGRAM],
                 [ax_cv_check_glew_libglew="${ax_try_lib}"; break])
done

AS_IF([test "X$ax_cv_check_glew_libglew" = Xno -a "X$no_x" = Xyes],
[LIBS='-framework GLEW'
AC_LINK_IFELSE([AX_CHECK_GLEW_PROGRAM],
               [ax_cv_check_glew_libglew="$LIBS"])])

CPPFLAGS="${ax_save_CPPFLAGS}"
LIBS="${ax_save_LIBS}"
AC_LANG_POP(C)])

AS_IF([test "X$ax_cv_check_glew_libglew" = Xno],
      [no_glew="yes"; GLEW_CFLAGS=""; GLEW_LIBS=""],
      [GLEW_LIBS="${ax_cv_check_glew_libglew} ${GLEW_LIBS}"])

AC_SUBST([GLEW_CFLAGS])
AC_SUBST([GLEW_LIBS])
])dnl
