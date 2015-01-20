# ax_vtk.m4: A m4 macro to detect and configure Vtk library.
#
# loosely adapted from the original vtk.m4 by Francesco Montorsi found on 
# the following web page : http://www.itk.org/Wiki/VTK_Autoconf
#
# Pierre Kestener.

#
# SYNOPSIS
#       AX_VTK([minimun-version], [action-if-found], [action-if-not-found])
#
# DESCRIPTION
#       Adds the --with-vtk=path option to the configure options
#       Adds the --with-vtk-version=path option to the configure options
#       Defines variables :
#        - want_vtk  : yes or no
#        - with_vtk  : PATH to vtk library
#
# NOTE: [minimum-version] must be in the form [X.Y.Z]
#
AC_DEFUN([AX_VTK],
[
	AC_ARG_WITH([vtk],
		[AC_HELP_STRING([--with-vtk],
		[The prefix where VTK is installed (default is /usr)])],
              	[
			if test "$withval" = "no"
			then
				want_vtk="no"
			elif test "$withval" = "yes"
			then
				want_vtk="yes"
				# Note we will try to find vtk in some standard locations
			else
				with_vtk="$withval"
				want_vtk="yes"
			fi
	      	],
          	[
			want_vtk="yes"
	      	]
	)

	AC_ARG_WITH([vtk-version],
		[AC_HELP_STRING([--with-vtk-version],
    		[VTK's include directory name is vtk-suffix, e.g. vtk-5.2/. What's the suffix? (Default -5.2)])],
    		[vtk_suffix=$withval],
		[vtk_suffix="-5.2"])

	# try to find a valid vtk location (and set have_vtk to yes in that case)
	have_vtk="no"
	if test "$want_vtk" = "yes"
	then
		# if the configure switch is used to define the VTK path
		if test -n "$with_vtk"
		then
			# a path to vtk installation location was provided
			VTK_PREFIX="${with_vtk}"
			if test -f "${VTK_PREFIX}/include/vtk${vtk_suffix}/vtkCommonInstantiator.h"
			then
				VTK_INCLUDE_PATH="${VTK_PREFIX}/include/vtk${vtk_suffix}"
				VTK_LIBRARY_PATH="${VTK_PREFIX}/lib/vtk${vtk_suffix}"
				have_vtk="yes"
			fi
		else
			#otherwise, try to find VTK in some standard locations
			for VTK_PREFIX_TMP in /usr /usr/local
			do
				if test -f "${VTK_PREFIX_TMP}/include/vtk${vtk_suffix}/vtkCommonInstantiator.h"
				then
					VTK_INCLUDE_PATH="${VTK_PREFIX_TMP}/include/vtk$vtk_suffix"
					VTK_LIBRARY_PATH="${VTK_PREFIX_TMP}/lib/vtk$vtk_suffix"
					have_vtk="yes"
					break;
				fi	
			done
		fi

		if test "$have_vtk" = "yes"
		then
			VTK_CPPFLAGS="-I${VTK_INCLUDE_PATH}"
			VTK_LDFLAGS="-L${VTK_LIBRARY_PATH}"
			VTK_LIBS="-lvtkCommon -lvtkDICOMParser -lvtkFiltering -lvtkftgl -lvtkGraphics -lvtkHybrid -lvtkImaging -lvtkIO -lvtkRendering -lvtkParallel "

			AC_MSG_CHECKING([for vtk library])

		      	# now, eventually check version
      			if [[ -n "$1" ]]; 
			then
			   #
        		   # A version was specified... 
			   # parse the version string in $1
			   
			   # The version of VTK that we need:
        		   maj=`echo $1 | sed 's/\([[0-9]]*\).\([[0-9]]*\).\([[0-9]]*\)/\1/'`
			   min=`echo $1 | sed 's/\([[0-9]]*\).\([[0-9]]*\).\([[0-9]]*\)/\2/'`
			   rel=`echo $1 | sed 's/\([[0-9]]*\).\([[0-9]]*\).\([[0-9]]*\)/\3/'`
			   AC_MSG_CHECKING([if VTK version is at least $maj.$min.$rel])
			   
 			   # Compare required version of VTK against installed version:
			   #
			   # Note that in order to be able to compile the following
			   # test program, we need to add to the current flags, 
			   # the VTK settings...
			   OLD_CFLAGS=$CFLAGS
			   OLD_CXXFLAGS=$CXXFLAGS
			   OLD_LDFLAGS=$LDFLAGS
			   OLd_LIBS=$LIBS
			   CFLAGS="$VTK_CPPFLAGS $CFLAGS"
			   CXXFLAGS="$VTK_CPPFLAGS $CXXFLAGS"
			   LDFLAGS="$VTK_LDFLAGS $LDFLAGS"
			   LIBS="$VTK_LIBS $LIBS"
        		   #
			   # check if the installed VTK is greater or not
			   AC_COMPILE_IFELSE([AC_LANG_PROGRAM(
				[
				#include <vtkConfigure.h>
				#include <stdio.h>
				],
              			[
                		printf("VTK version is: %d.%d.%d", VTK_MAJOR_VERSION, VTK_MINOR_VERSION, VTK_BUILD_VERSION);
				#if VTK_MAJOR_VERSION < $maj
                		#error Installed VTK is too old !
                		#endif
                		#if VTK_MINOR_VERSION < $min
                		#error Installed VTK is too old !
                		#endif
                		#if VTK_BUILD_VERSION < $rel
                		#error Installed VTK is too old !
                		#endif
              			])
        			], 
				[vtkVersion="OK"])
			    #
			    # restore all flags without VTK values
			    CFLAGS=$OLD_CFLAGS
			    CXXFLAGS=$OLD_CXXFLAGS
			    LDFLAGS=$OLD_LDFLAGS
			    LIBS=$OLD_LIBS
        		    #
			    # Execute $2 if version is ok, 
			    # otherwise execute $3
        		    if [[ "$vtkVersion" = "OK" ]]; then
			       	AC_MSG_RESULT([yes])
          			$2
        		    else
				AC_MSG_RESULT([no])
          			$3
        		    fi
        		    #
			else
				# A target version number was not provided... execute $2 unconditionally
				AC_MSG_RESULT([yes])
				$2
			fi
		fi

		if test "$have_vtk" = "yes"
		then
			# these are the VTK libraries of a default build
			AC_SUBST(VTK_CPPFLAGS)
			AC_SUBST(VTK_LDFLAGS)
			AC_SUBST(VTK_LIBS)
			AC_DEFINE(HAVE_VTK, 1, [Define if you have Vtk library])
		fi

	else # !want_vtk
	     have_vtk="no"
	fi
])# AX_VTK


