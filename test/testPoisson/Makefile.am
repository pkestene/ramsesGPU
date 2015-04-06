AUTOMAKE_OPTIONS = subdir-objects

# clean generated files
MAINTAINERCLEANFILES = $(srcdir)/Makefile.in

#CXX=@MPICXX@
AM_CPPFLAGS = -Wno-unused-parameter -Wno-unused-function

bin_PROGRAMS = testPoissonCpuFFTW2d testPoissonCpuFFTW3d

testPoissonCpuFFTW2d_SOURCES  = testPoissonCpuFFTW2d.cpp
testPoissonCpuFFTW2d_CPPFLAGS = $(AM_CPPFLAGS) -I$(top_srcdir)/src/utils/cnpy -I$(top_srcdir)/src/hydro 
testPoissonCpuFFTW2d_LDFLAGS  = $(AM_LDFLAGS) $(FFTW3_LDFLAGS) ../../src/utils/cnpy/libCNpy.la

testPoissonCpuFFTW3d_SOURCES  = testPoissonCpuFFTW3d.cpp
testPoissonCpuFFTW3d_CPPFLAGS = $(AM_CPPFLAGS) -I$(top_srcdir)/src/utils/cnpy -I$(top_srcdir)/src/hydro 
testPoissonCpuFFTW3d_LDFLAGS  = $(AM_LDFLAGS) $(FFTW3_LDFLAGS) ../../src/utils/cnpy/libCNpy.la
