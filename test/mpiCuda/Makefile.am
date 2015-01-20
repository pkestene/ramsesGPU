# also note that SUFFIXES must be set with .cu before including cuda.am
# ortherwise this Makefile is not usable
SUFFIXES = .cu
include $(top_srcdir)/am/cuda.am

# clean generated files
MAINTAINERCLEANFILES = $(srcdir)/Makefile.in 

AUTOMAKE_OPTIONS = subdir-objects

AM_CXXFLAGS = $(CUDA_CFLAGS)
AM_CFLAGS   = $(CUDA_CFLAGS)

NVCCFLAGS  += --compiler-options -I$(srcdir)
NVCCFLAGS  += $(addprefix --compiler-options , $(MPI_CXXFLAGS) )
NVCCFLAGS  += --compiler-options -I$(top_srcdir)/src/utils/mpiUtils
NVCCFLAGS  += --compiler-options -I$(top_srcdir)/src/hydro

# ===================================================
# PROGRAMS
# ===================================================
bin_PROGRAMS  = testHelloMpiCuda testBorderBufferCuda testBorderBufferCuda2

#
# testHelloMpiCuda
#
testHelloMpiCuda_SOURCES  = testHelloMpiCuda.cu
# add dummy C++ source to cause C++ linking.
nodist_EXTRA_testHelloMpiCuda_SOURCES = dummy.cxx
testHelloMpiCuda_CPPFLAGS = $(AM_CPPFLAGS) $(MPI_CXXFLAGS) -I$(top_srcdir)/src/utils/mpiUtils
testHelloMpiCuda_LDFLAGS  = $(AM_LDFLAGS)  $(CUDA_LIBS) ../../src/utils/mpiUtils/libMpiUtils.la $(MPI_LDFLAGS) 

#
# testBorderBufferCuda
#
testBorderBufferCuda_SOURCES = testBorderBufferCuda.cu
# add dummy C++ source to cause C++ linking.
nodist_EXTRA_testBorderBufferCuda_SOURCES = dummy.cxx
testBorderBufferCuda_CPPFLAGS = $(AM_CPPFLAGS) $(MPI_CXXFLAGS) -I$(top_srcdir)/src/hydro -I$(top_srcdir)/src/utils/mpiUtils
testBorderBufferCuda_LDFLAGS  = $(AM_LDFLAGS)  $(MPI_LDFLAGS) $(CUDA_LIBS) ../../src/hydro/libhydroGpu.la ../../src/utils/mpiUtils/libMpiUtils.la

#
# testBorderBufferCuda2
#
testBorderBufferCuda2_SOURCES = testBorderBufferCuda2.cu
# add dummy C++ source to cause C++ linking.
nodist_EXTRA_testBorderBufferCuda2_SOURCES = dummy.cxx
testBorderBufferCuda2_CPPFLAGS = $(AM_CPPFLAGS) $(MPI_CXXFLAGS) -I$(top_srcdir)/src/hydro -I$(top_srcdir)/src/utils/mpiUtils
testBorderBufferCuda2_LDFLAGS  = $(AM_LDFLAGS)  $(MPI_LDFLAGS) $(CUDA_LIBS) ../../src/hydro/libhydroGpu.la ../../src/utils/mpiUtils/libMpiUtils.la

