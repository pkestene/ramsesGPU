# ------------------------------------------------------------------------------
# Package setup
# ------------------------------------------------------------------------------
AC_INIT([RAMSES-GPU], [1.3.0], [pierre.kestener@cea.fr])

AM_INIT_AUTOMAKE([-Wall -Werror foreign -Wno-portability])

# ------------------------------------------------------------------------------
# Default flags reinitialization
# ------------------------------------------------------------------------------
if test -z "$FCFLAGS"
then
    FCFLAGS="-Wall"
fi

if test -z "$CFLAGS"
then
   CFLAGS=""
fi

if test -z "$CXXFLAGS"
then
   CXXFLAGS="-Wall"
fi

# ------------------------------------------------------------------------------
# Dummy headline formater
# ------------------------------------------------------------------------------
AC_DEFUN([CONFIGURE_HEADLINE],
[
        echo; echo "+++ $1 +++"
])


# ------------------------------------------------------------------------------
# Checks for tool support
# ------------------------------------------------------------------------------
AC_PROG_FC
AC_PROG_CXX
AC_PROG_LIBTOOL

# some compiler vendor specific flags (intel icc does not understand -Wextra)
AX_COMPILER_VENDOR
case "${ax_cv_c_compiler_vendor}" in
gnu)
	CXXFLAGS+=" -Wall -Wextra"
	;;
intel)
	# disable remark 981: operands are evaluated in unspecified order
	CXXFLAGS+=" -Wall -wd981"
	;;
*)
	;;
esac

# ------------------------------------------------------------------------------
# Checks for SVN svnversion support
# adapted from http://mhoenicka.de/system-cgi/blog/index.php?itemid=999
# ------------------------------------------------------------------------------
AC_PATH_PROG(svnversioncommand, svnversion)

# use svnversion to record the current repository revision only if
# subversion is installed and we are in a working copy
#
# NOTICE : here we don't need to worry about language
# configure script forces the use of variable LANGUAGE=C
#
if test "X$svnversioncommand" = "X" || test `$svnversioncommand -n $srcdir` = "exported"; then
   MYSVNVERSION="if test ! -f svn_dist_version; then echo -n 'Unknow version (sources obtained for svn export)' ; else cat svn_dist_version; fi"
else
   # we need an absolute path here since srcdir is relative to top build directory
   MYSVNVERSION="svnversion -n `pwd`/$srcdir"
fi

SVN_REVISION=`$MYSVNVERSION`
AC_SUBST(SVN_REVISION)

# the conditional "HAVE_SVN_WC" will be used in src/Makefile.am to make sure
# the target appears in Makefile only if we are building a Subversion working copy. 
AM_CONDITIONAL(HAVE_SVN_WC, test "X$svnversioncommand" != "X" && test `$svnversioncommand -n $srcdir` != "exported")

AC_SUBST([MYSVNVERSION])


# ------------------------------------------------------------------------------
# Checks for Fortran support
# ------------------------------------------------------------------------------
AC_LANG([Fortran])
AC_FC_LIBRARY_LDFLAGS

# ------------------------------------------------------------------------------
# Checks for C/C++ support
# ------------------------------------------------------------------------------
AC_LANG([C])
AC_LANG([C++])

# ------------------------------------------------------------------------------
# Enable silent build rules by default
# ------------------------------------------------------------------------------
m4_ifdef([AM_SILENT_RULES],[AM_SILENT_RULES([yes])],[AC_SUBST([AM_DEFAULT_VERBOSITY], [1])])

# ------------------------------------------------------------------------------
# OpenMP support
# ------------------------------------------------------------------------------
AC_ARG_ENABLE([openmp],
    [AS_HELP_STRING([--enable-openmp],[use OpenMP directives for parallelism])],
    [case "${enableval}" in
       yes) enable_openmp=yes ;;
       no)  enable_openmp=no ;;
       *)   AC_MSG_ERROR([bad value ${enableval} for --enable-openmp]) ;;
     esac], 
     enable_openmp=no
)
if test x$enable_openmp = xyes
then
    AC_DEFINE(HAVE_OPENMP,1,[Define to 1 to enable OpenMP.])
    AX_OPENMP([THREADLIBS=" "
     AC_DEFINE(USING_OPENMP_THREADS, 1, 
       [Define to 1 if there is a valid OpenMP flag and the compiler supports it.])
     CXXFLAGS="$CXXFLAGS $OPENMP_CXXFLAGS"],
     [AC_MSG_ERROR([Unable to determine the appropriate OpenMP flag for your compiler.
         Please set the CFLAGS variable manually for OpenMP support.])])
fi


# ------------------------------------------------------------------------------
# Provides debug and release compilation modes
# ------------------------------------------------------------------------------
AC_ARG_ENABLE([debug],
   [  --enable-debug    Turn on debugging],
   [case "${enableval}" in
      yes) DEBUG=true ;;
      no)  DEBUG=false ;;
      *)   AC_MSG_ERROR([bad value ${enableval} for --enable-debug]) ;;
   esac],
   [DEBUG=false]
)
if test x$DEBUG = xtrue
then
    FCFLAGS="-g3 -O0 ${FCFLAGS}"
    CXXFLAGS="-g3 -O0 -D_GLIBCXX_DEBUG ${CXXFLAGS}"
    CFLAGS="-g3 -O0 -D_GLIBCXX_DEBUG ${CFLAGS}"
else
    FCFLAGS="-O3 ${FCFLAGS}"
    CXXFLAGS="-O3 ${CXXFLAGS}"
    CFLAGS="-O3 ${CFLAGS}"
fi

# ------------------------------------------------------------------------------
# Turn on gprof flags
# ------------------------------------------------------------------------------
AC_ARG_ENABLE([gprof],
	[  --enable-gprof    Turn on profiling with gprof (pass gnu or intel as argument)],
	[case "${enableval}" in
	      gnu | yes) 
	      	   gprof_enabled=true 
		   gprof_compiler="gnu compiler"
		   gprof_flags="-pg"
		   ;;
	      intel)
		   gprof_enabled=true
		   gprof_compiler="intel compiler"
		   gprof_flags="-p"
		   ;;
	      no)  gprof_enabled=false 
	      	   gprof_compiler="none"
		   gprof_flags=""
		   ;;
	      *) AC_MSG_ERROR([bad value ${enableval} for --enable-gprof (only yes, gnu or intel are possible)]) ;;
	 esac],
	 [gprof_enabled=false
	  gprof_compiler="none"
	  gprof_flags=""
	 ]	      
)
if test x$gprof_enabled = xtrue
then
	CXXFLAGS="${CXXFLAGS} ${gprof_flags}"
	CPPFLAGS="${CPPFLAGS} ${gprof_flags}"
	LDFLAGS="${LDFLAGS} ${gprof_flags}"
fi

# ------------------------------------------------------------------------------
# Turn on timing measurements
# ------------------------------------------------------------------------------
AC_ARG_ENABLE([timing],
	[  --enable-timing    Turn on timing measurement and report],
	[case "${enableval}" in
	      yes) 
	      	   timing_enabled=true 
		   ;;
	      no)  timing_enabled=false 
		   ;;
	      *) AC_MSG_ERROR([bad value ${enableval} for --enable-timing]) ;;
	 esac],
	 [timing_enabled=false
	 ]	      
)
AM_CONDITIONAL(DO_TIMING, test "$timing_enabled" = "true")

# ------------------------------------------------------------------------------
# Turn on double precision
# ------------------------------------------------------------------------------
AC_ARG_ENABLE([double],
	[  --enable-double    Turn on double precision computation],
	[case "${enableval}" in
	      yes) DOUBLE=yes ;;
	      no)  DOUBLE=no ;;
	      *)   AC_MSG_ERROR([bad value ${enableval} for --enable-double]) ;;
	      esac],
   	[DOUBLE=no]
)
AM_CONDITIONAL(USE_DOUBLE_PRECISION, test "$DOUBLE" = "yes")

# ------------------------------------------------------------------------------
# Turn on double precision
# ------------------------------------------------------------------------------
AC_ARG_ENABLE([mixed-precision],
	[  --enable-mixed-precision    Turn on mixed precision computation],
	[case "${enableval}" in
	      yes) MIXED=yes ;;
	      no)  MIXED=no ;;
	      *)   AC_MSG_ERROR([bad value ${enableval} for --enable-mixed]) ;;
	      esac],
   	[MIXED=no]
)
AM_CONDITIONAL(USE_MIXED_PRECISION, test "$MIXED" = "yes")

# ------------------------------------------------------------------------------
# Detects CUDA
# ------------------------------------------------------------------------------
CONFIGURE_HEADLINE([ CUDA support ])
AX_CUDA

# ------------------------------------------------------------------------------
# Nvcc flags setup (remember that default NVCCFLAGS are set by AX_CUDA macro)
# ------------------------------------------------------------------------------
if test x$DEBUG = xtrue
then
    NVCCFLAGS+=" -g -G"
else
    NVCCFLAGS+=" -O3"
fi

if test x$gprof_enabled = xtrue
then
    NVCCFLAGS+=" --compiler-options ${gprof_flags}"
fi

# ------------------------------------------------------------------------------
#    Check for lib GraphicsMagick++
# ------------------------------------------------------------------------------
CONFIGURE_HEADLINE([ Graphics Magick support ])
PKG_CHECK_MODULES(GM, [GraphicsMagick++ >= 1.1], 
		      [AC_DEFINE([HAVE_GRAPHICS_MAGICK_PP],[1],[GraphicsMagick++ available])
		      enable_gm="yes"],
		      [AC_MSG_WARN("GraphicsMagick++ disabled")
		      enable_gm="no"])
AM_CONDITIONAL(HAVE_GM, test -n "$GM_LIBS")
AC_SUBST([GM_CFLAGS])
AC_SUBST([GM_LIBS])

# ------------------------------------------------------------------------------
#    Check for lib Vtk
# ------------------------------------------------------------------------------
CONFIGURE_HEADLINE([ VTK support ])
AX_VTK(5.2.0)
AM_CONDITIONAL(HAVE_VTK, test "$have_vtk" = "yes")

# ------------------------------------------------------------------------------
#    Check for lib Hdf5
# ------------------------------------------------------------------------------
CONFIGURE_HEADLINE([ HDF5 support ])
AX_LIB_HDF5()
AM_CONDITIONAL(HAVE_HDF5, test "$with_hdf5" = "yes")
AM_CONDITIONAL(HAVE_HDF5_PARALLEL, test "$HDF5_PARALLEL_ENABLED" = "yes")

# ------------------------------------------------------------------------------
#    Check for lib NetCDF4
# ------------------------------------------------------------------------------
CONFIGURE_HEADLINE([ NetCDF4 support ])
AX_LIB_NETCDF4()
AM_CONDITIONAL(HAVE_NETCDF4, test "$with_netcdf4" = "yes")

# ------------------------------------------------------------------------------
#    Check for lib Parallel-NetCDF
# ------------------------------------------------------------------------------
CONFIGURE_HEADLINE([ Parallel-NetCDF support ])
AX_LIB_PNETCDF()
AM_CONDITIONAL(HAVE_PNETCDF, test "$with_pnetcdf" = "yes")

# ------------------------------------------------------------------------------
#    Check for lib fftw
# ------------------------------------------------------------------------------
CONFIGURE_HEADLINE([ FFTW3 support ])
AX_LIB_FFTW3_DOUBLE()
AM_CONDITIONAL(HAVE_FFTW3_MPI, test "$with_fftw3_mpi" = "yes")

# ------------------------------------------------------------------------------
#    Check for lib MPI (OpenMPI or MPICH)
# ------------------------------------------------------------------------------
CONFIGURE_HEADLINE([ MPI compiler ])
AC_ARG_ENABLE(mpi, [AC_HELP_STRING([--enable-mpi],[MPI library required])], enable_mpi=$enableval, enable_mpi=no)

if test "$enable_mpi" = "yes"; then
   ACX_MPI([],[AC_MSG_ERROR([could not find mpi library for --enable-mpi])])
   AC_CHECK_PROG(MPIRUN, mpirun, mpirun)
   AC_SUBST(MPIRUN)
else
   AC_MSG_NOTICE([MPI is currently disabled; use option --enable-mpi to enable])
fi
AM_CONDITIONAL(USE_MPI, test "$enable_mpi" = "yes")

# try to find if we are using OpenMPI / MPICH by looking inside mpi.h
AS_IF([test "$enable_mpi" = "yes"],
	[
		sav_CXX=$CXX
		sav_CXXFLAGS=$CXXFLAGS
		CXX=$MPICXX
		CXXFLAGS="$CXXFLAGS"
		AC_CHECK_DECL([OPEN_MPI], [mpi_vendor="OpenMPI"],
		[], [#include "mpi.h"])
		AC_CHECK_DECL([MPICH2], [mpi_vendor="MPICH"],
		[], [#include "mpi.h"])
		AC_CHECK_DECL([MPICH], [mpi_vendor="MVAPICH2"],
		[], [#include "mpi.h"])
		CXX=$sav_CXX
		CXXFLAGS=$sav_CXXFLAGS
	])

#
# try to set MPI_CXXFLAGS and MPI_LDFLAGS
#
MPI_CXXFLAGS=
MPI_LDFLAGS=
if test "$enable_mpi" = "yes" 
then
    if test "$mpi_vendor" = "OpenMPI"
    then
	MPI_CXXFLAGS=`$MPICXX --showme:compile`
     	MPI_LDFLAGS=`$MPICXX --showme:link`
     	AC_MSG_NOTICE([$mpi_vendor found])
     	AC_MSG_NOTICE([MPI_CXXFLAGS=$MPI_CXXFLAGS])
     	AC_MSG_NOTICE([MPI_LDFLAGS=$MPI_LDFLAGS])
    elif test "$mpi_vendor" = "MPICH" -o "$mpi_vendor" = "MVAPICH2"
    then
	# first grab CXXFLAGS (awk script is used to remove CXX at front)
     	tmp=`$MPICXX -compile-info | awk '{$1=""; print $0 }'`
     	MPI_CXXFLAGS=
     	for i in $tmp
     	do 
     	   case $i in 
	   	-[[DIUbi]]*)
		  MPI_CXXFLAGS="$MPI_CXXFLAGS $i"
             	  ;;
	   esac
     	done
     	# second grab LDFLAGS (awk script is used to remove CXX at front)
     	tmp=`$MPICXX -link-info | awk '{$1=""; print $0 }'`
     	for i in $tmp
     	do 
     	   case $i in 
     	      	[[\\/]]*.a | ?:[[\\/]]*.a | -[[lLRu]]* | -Wl* )
     	      	  MPI_LDFLAGS="$MPI_LDFLAGS $i"
     	       	  ;;
           esac
     	done
	AC_MSG_NOTICE([$mpi_vendor found])
     	AC_MSG_NOTICE([MPI_CXXFLAGS=$MPI_CXXFLAGS])
     	AC_MSG_NOTICE([MPI_LDFLAGS=$MPI_LDFLAGS])
    else
     AC_MSG_WARN([Neither OpenMPI / MPICH / MVAPICH2 have been recognized...])
    fi
fi

AC_SUBST(MPI_CXXFLAGS)
AC_SUBST(MPI_LDFLAGS)

# ------------------------------------------------------------------------------
#    Check for BOOST / MPI
# ------------------------------------------------------------------------------
dnl CONFIGURE_HEADLINE([ BOOST and MPI ])
dnl AX_BOOST_BASE([1.40.0])

dnl if test x$want_boost = xyes; then
dnl    AX_BOOST_SERIALIZATION

dnl    # look for boost_mpi
dnl    sav_CXX=$CXX
dnl    sav_CXXFLAGS=$CXXFLAGS
dnl    CXX=$MPICXX
dnl    CXXFLAGS="$CXXFLAGS"
dnl    AX_BOOST_MPI
dnl    CXX=$sav_CXX
dnl    CXXFLAGS=$sav_CXXFLAGS
dnl fi

dnl if test "x$BOOST_MPI_LIB" != "x";
dnl then
dnl 	use_boost_mpi=yes
dnl else
dnl 	use_boost_mpi=no
dnl fi
dnl AM_CONDITIONAL(USE_MPI_BOOST, test x"$use_boost_mpi" = x"yes")

# ------------------------------------------------------------------------------
# PAPI - performance measurements
# ------------------------------------------------------------------------------
CONFIGURE_HEADLINE([ PAPI support ])
AX_LIB_PAPI()
AM_CONDITIONAL(HAVE_PAPI, test "$with_papi" = "yes")

# ------------------------------------------------------------------------------
# MPI and DOUBLE PRECISION
# ------------------------------------------------------------------------------
AM_CONDITIONAL(USE_MPI_AND_DOUBLE_PRECISION, test "$enable_mpi" = "yes" && test "$DOUBLE" = "yes")

# ------------------------------------------------------------------------------
#    Check for libraries GLUT/GLEW (to make a simple GUI)
# ------------------------------------------------------------------------------
CONFIGURE_HEADLINE([ GLUT/GLEW ])
AX_CHECK_GLUT
AX_CHECK_GLEW
GLUT_ENABLE=yes
if test "x$no_glut" = "xyes"; then
   GLUT_ENABLE=no
fi
if test "x$no_glew" = "xyes"; then
   GLUT_ENABLE=no
fi

# define flag to enable / disable glut gui
AC_ARG_ENABLE([glutgui],
   AS_HELP_STRING([--enable-glutgui], [Turn on GLUT GUI]),
   [case "${enableval}" in
      yes) enable_glutgui=yes ;;
      no)  enable_glutgui=no ;;
      *)   AC_MSG_ERROR([bad value ${enableval} for --enable-glutgui]) ;;
   esac],
   [enable_glutgui=yes]
)

GLUT_AM_COND=no
if test "$GLUT_ENABLE" = "yes" && test "$enable_glutgui" = "yes"
then
	GLUT_AM_COND=yes
fi
AM_CONDITIONAL(HAVE_GLUT, test "$GLUT_AM_COND" = "yes")


# ------------------------------------------------------------------------------
#    Enable/disable Qt Gui building
# ------------------------------------------------------------------------------
CONFIGURE_HEADLINE([ QT ])
AX_QTGUI([opengl])

# ------------------------------------------------------------------------------
# DOC OPTIONS
# ------------------------------------------------------------------------------
# the default behavior is to build documentation if doxygen is found
DX_HTML_FEATURE(ON)
DX_CHM_FEATURE(OFF)
DX_CHI_FEATURE(OFF)
DX_MAN_FEATURE(OFF)
DX_RTF_FEATURE(OFF)
DX_XML_FEATURE(OFF)
DX_PDF_FEATURE(OFF)
DX_PS_FEATURE(OFF)
DX_INIT_DOXYGEN([$PACKAGE_NAME], [doc/Doxyfile.cfg])


# ------------------------------------------------------------------------------
# Declares output files
# ------------------------------------------------------------------------------
AC_CONFIG_FILES([
	Makefile 
	src/Makefile 
	src/hydro/Makefile
	src/utils/Makefile
	src/utils/monitoring/Makefile
	src/utils/config/Makefile
	src/utils/mpiUtils/Makefile
	src/utils/IO/Makefile
	src/utils/cnpy/Makefile
	src/qtGui/qtHelloWorld/Makefile
	src/analysis/powerSpectrum/Makefile
	src/analysis/readSlice/Makefile
	src/analysis/structureFunctions/Makefile
	test/Makefile
	test/test_run.sh
	test/computeL2relatif.py
	test/plotXsm.py
	test/mpiBasic/Makefile
	test/mpiHydro/Makefile
	test/mpiCuda/Makefile
	test/testPoisson/Makefile
	doc/Doxyfile.cfg
	doc/simu.html])
AC_OUTPUT

AC_MSG_RESULT([
	--------------------------------------------------
	--------------------------------------------------
	$PACKAGE_NAME is governed by the CeCILL  license http://www.cecill.info
	The fact that you are presently reading this means that you have had
	knowledge of the CeCILL license and that you accept its terms.

	--------------------------------------------------
	Configuration summary for $PACKAGE_NAME ($VERSION)
        --------------------------------------------------

	* Installation prefix    : $prefix
	* SVN revision           : $SVN_REVISION (as set at the last configure run)
	                           Note that only the run time svn version matters
				   as printed by C routine defined in svn_version.c
	* Build debug            : $DEBUG
	* gprof profiler enabled : $gprof_enabled ($gprof_compiler)
	* timing  report enabled : $timing_enabled
	* Double enabled         : $DOUBLE
	* mixed precision enabled: $MIXED

	* MPI    enabled         : $enable_mpi ($mpi_vendor)
	* OpenMP enabled	 : $enable_openmp (openmp cxxflags: $OPENMP_CXXFLAGS)

	* GPU/CUDA enabled       : $want_cuda
	* CUDA tools version     : $NVCC_VERSION (NVCCFLAGS = $NVCCFLAGS)
	* GLUT-based GUI enable  : $GLUT_AM_COND
	* QT-based GUI enable    : $enable_qtgui
	* VTK enabled            : $have_vtk ($VTK_LIBRARY_PATH)
	*                        : vtk libs: $VTK_LIBS
	* HDF5 enabled           : $with_hdf5 ($HDF5_VERSION)
	* NETCDF4 enabled        : $with_netcdf4 ($NETCDF_VERSION)
	* Parallel-NetCDF        : $with_pnetcdf ($PNETCDF_VERSION)
	* FFTW3-MPI enabled      : $with_fftw3_mpi (if no try to set env variable FFTW3_ROOT or FFTW3_LIBDIR)
	* GraphicsMagick++       : $enable_gm ($GM_LIBS)
	* papi perf measurements : $with_papi ($PAPI_PATH)
])
