AUTOMAKE_OPTIONS = subdir-objects

# clean generated files
MAINTAINERCLEANFILES = $(srcdir)/Makefile.in

#CXX=@MPICXX@
AM_CPPFLAGS = -Wno-unused-parameter -Wno-unused-function

bin_PROGRAMS = testPoissonCpuFFTW

testPoissonCpuFFTW_SOURCES  = testPoissonCpuFFTW.cpp
testPoissonCpuFFTW_CPPFLAGS = $(AM_CPPFLAGS) -I$(top_srcdir)/src/utils/cnpy -I$(top_srcdir)/src/hydro 
testPoissonCpuFFTW_LDFLAGS  = $(AM_LDFLAGS) $(FFTW3_LDFLAGS) ../../src/utils/cnpy/libCNpy.la
