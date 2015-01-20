SUFFIXES = .cu
include $(top_srcdir)/am/cuda.am
NVCCFLAGS += --ptxas-options -v
NVCCFLAGS += --compiler-options -I$(srcdir)

# clean generated files
MAINTAINERCLEANFILES = $(srcdir)/Makefile.in

AUTOMAKE_OPTIONS = subdir-objects
SUBDIRS = .

# ===================================================
# Some convenience library
# ===================================================

noinst_LTLIBRARIES = libMonitoring.la
if USE_CUDA
noinst_LTLIBRARIES += libMonitoringGpu.la
endif

libMonitoring_la_SOURCES = \
	date.h \
	measure_time.h \
	CudaTimer.h \
	Timer.h \
	Timer.cpp
if HAVE_PAPI
libMonitoring_la_SOURCES += PapiInfo.h PapiInfo.cpp
endif
libMonitoring_la_CPPFLAGS = $(AM_CPPFLAGS)
if HAVE_PAPI
libMonitoring_la_CPPFLAGS += $(PAPI_CPPFLAGS)
endif

if USE_CUDA
# libMonitoring parsed/compiled with nvcc
BUILT_SOURCES = Timer.cu
CLEANFILES    = Timer.cu
libMonitoringGpu_la_SOURCES = Timer.h 
nodist_libMonitoringGpu_la_SOURCES = Timer.cu
nodist_EXTRA_libMonitoringGpu_la_SOURCES = dummy.cpp
libMonitoringGpu_la_CPPFLAGS = $(AM_CPPFLAGS) $(CUDA_CFLAGS)
endif


#
# Test app for PapiInfo class
#
if HAVE_PAPI
bin_PROGRAMS = PapiInfoTest
PapiInfoTest_SOURCES = PapiInfoTest.cpp
PapiInfoTest_CPPFLAGS = $(PAPI_CPPFLAGS)
PapiInfoTest_LDFLAGS = libMonitoring.la $(PAPI_LDFLAGS)
endif
