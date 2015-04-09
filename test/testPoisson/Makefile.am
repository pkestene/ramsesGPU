# take care that autotroll.am 
# already defines the following variables
# SUFFIXES, DISTCLEANFILES and CLEANFILES
include $(top_srcdir)/am/autotroll.am
# also note that SUFFIXES must be set with .cu before including cuda.am
# ortherwise this Makefile is not usable
SUFFIXES += .cu
include $(top_srcdir)/am/cuda.am

AUTOMAKE_OPTIONS = subdir-objects

# clean generated files
MAINTAINERCLEANFILES = $(srcdir)/Makefile.in

#CXX=@MPICXX@
AM_CXXFLAGS = $(CUDA_CFLAGS)
AM_CFLAGS   = $(CUDA_CFLAGS)
AM_CPPFLAGS = -Wno-unused-parameter -Wno-unused-function

NVCCFLAGS  += --ptxas-options -v
NVCCFLAGS  += --compiler-options -I$(srcdir)
NVCCFLAGS  += $(addprefix --compiler-options , $(MPI_CXXFLAGS) )
NVCCFLAGS  += --compiler-options -I$(top_srcdir)/src/utils/mpiUtils
NVCCFLAGS  += --compiler-options -I$(top_srcdir)/src/hydro
NVCCFLAGS  += --compiler-options -I$(top_srcdir)/src/utils/cnpy

# ===================================================
# PROGRAMS
# ===================================================
bin_PROGRAMS = testPoissonCpuFFTW2d testPoissonCpuFFTW3d
if USE_DOUBLE_PRECISION
bin_PROGRAMS += testPoissonCpuFFTW2d_double testPoissonCpuFFTW3d_double
endif

if USE_CUDA
bin_PROGRAMS += testPoissonGpuCuFFT2d testPoissonGpuCuFFT3d
if USE_DOUBLE_PRECISION
bin_PROGRAMS += testPoissonGpuCuFFT2d_double testPoissonGpuCuFFT3d_double
endif
endif




#
# testPoissonCpuFFTW2d
#
testPoissonCpuFFTW2d_SOURCES  = testPoissonCpuFFTW2d.cpp
testPoissonCpuFFTW2d_CPPFLAGS = -DUSE_FLOAT $(AM_CPPFLAGS) -I$(top_srcdir)/src/utils/cnpy -I$(top_srcdir)/src/hydro 
testPoissonCpuFFTW2d_LDFLAGS  = $(AM_LDFLAGS) $(FFTW3_LDFLAGS) ../../src/utils/cnpy/libCNpy.la

#
# testPoissonCpuFFTW3d
#
testPoissonCpuFFTW3d_SOURCES  = testPoissonCpuFFTW3d.cpp
testPoissonCpuFFTW3d_CPPFLAGS = -DUSE_FLOAT $(AM_CPPFLAGS) -I$(top_srcdir)/src/utils/cnpy -I$(top_srcdir)/src/hydro 
testPoissonCpuFFTW3d_LDFLAGS  = $(AM_LDFLAGS) $(FFTW3_LDFLAGS) ../../src/utils/cnpy/libCNpy.la

if USE_DOUBLE_PRECISION
########################
### DOUBLE PRECISION ###
########################
#
# testPoissonCpuFFTW2d_double
#
testPoissonCpuFFTW2d_double_SOURCES  = testPoissonCpuFFTW2d.cpp
testPoissonCpuFFTW2d_double_CPPFLAGS = -DUSE_DOUBLE $(AM_CPPFLAGS) -I$(top_srcdir)/src/utils/cnpy -I$(top_srcdir)/src/hydro 
testPoissonCpuFFTW2d_double_LDFLAGS  = $(AM_LDFLAGS) $(FFTW3_LDFLAGS) ../../src/utils/cnpy/libCNpy.la

#
# testPoissonCpuFFTW3d_double
#
testPoissonCpuFFTW3d_double_SOURCES  = testPoissonCpuFFTW3d.cpp
testPoissonCpuFFTW3d_double_CPPFLAGS = -DUSE_DOUBLE $(AM_CPPFLAGS) -I$(top_srcdir)/src/utils/cnpy -I$(top_srcdir)/src/hydro 
testPoissonCpuFFTW3d_double_LDFLAGS  = $(AM_LDFLAGS) $(FFTW3_LDFLAGS) ../../src/utils/cnpy/libCNpy.la

endif

###############################################################################
## GPU
###############################################################################

if USE_CUDA
#
# testPoissonGpuCuFFT2d
#
testPoissonGpuCuFFT2d_SOURCES  = testPoissonGpuCuFFT2d.cu
# force CXX link
nodist_EXTRA_testPoissonGpuCuFFT2d_SOURCES = dummy.cpp
testPoissonGpuCuFFT2d_CPPFLAGS = -DUSE_FLOAT $(CUDA_CFLAGS) -I$(top_srcdir)/src/utils/cnpy -I$(top_srcdir)/src/hydro 
testPoissonGpuCuFFT2d_LDFLAGS  = $(CUDA_LIBS) -lcufftw -lcufft
testPoissonGpuCuFFT2d_LDADD    = ../../src/utils/cnpy/libCNpy.la

#
# testPoissonGpuCuFFT3d
#
testPoissonGpuCuFFT3d_SOURCES  = testPoissonGpuCuFFT3d.cu
# force CXX link
nodist_EXTRA_testPoissonGpuCuFFT3d_SOURCES = dummy.cpp
testPoissonGpuCuFFT3d_CPPFLAGS = -DUSE_FLOAT $(CUDA_CFLAGS) -I$(top_srcdir)/src/utils/cnpy -I$(top_srcdir)/src/hydro 
testPoissonGpuCuFFT3d_LDFLAGS  = $(CUDA_LIBS) -lcufftw -lcufft
testPoissonGpuCuFFT3d_LDADD    = ../../src/utils/cnpy/libCNpy.la

if USE_DOUBLE_PRECISION
########################
### DOUBLE PRECISION ###
########################
#
# testPoissonGpuCuFFT2d_double
#
testPoissonGpuCuFFT2d_double_SOURCES  = testPoissonGpuCuFFT2d.cu
# force CXX link
nodist_EXTRA_testPoissonGpuCuFFT2d_double_SOURCES = dummy.cpp
testPoissonGpuCuFFT2d_double_CPPFLAGS = -DUSE_DOUBLE $(CUDA_CFLAGS) -I$(top_srcdir)/src/utils/cnpy -I$(top_srcdir)/src/hydro 
testPoissonGpuCuFFT2d_double_LDFLAGS  = $(CUDA_LIBS) -lcufftw -lcufft
testPoissonGpuCuFFT2d_double_LDADD    = ../../src/utils/cnpy/libCNpy.la

#
# testPoissonGpuCuFFT3d_double
#
testPoissonGpuCuFFT3d_double_SOURCES  = testPoissonGpuCuFFT3d.cu
# force CXX link
nodist_EXTRA_testPoissonGpuCuFFT3d_double_SOURCES = dummy.cpp
testPoissonGpuCuFFT3d_double_CPPFLAGS = -DUSE_DOUBLE $(CUDA_CFLAGS) -I$(top_srcdir)/src/utils/cnpy -I$(top_srcdir)/src/hydro 
testPoissonGpuCuFFT3d_double_LDFLAGS  = $(CUDA_LIBS) -lcufftw -lcufft
testPoissonGpuCuFFT3d_double_LDADD    = ../../src/utils/cnpy/libCNpy.la

endif

endif
