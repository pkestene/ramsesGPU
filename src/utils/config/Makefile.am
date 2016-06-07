SUFFIXES = .cu
include $(top_srcdir)/am/cuda.am
NVCCFLAGS += --ptxas-options -v
NVCCFLAGS += --compiler-options -I$(srcdir)
NVCCFLAGS += --compiler-options -I$(srcdir)/inih

# clean generated files
MAINTAINERCLEANFILES = $(srcdir)/Makefile.in

AUTOMAKE_OPTIONS = subdir-objects
SUBDIRS = .

# ===================================================
# Some convenience library
# ===================================================

noinst_LTLIBRARIES = libIniConfig.la 
if USE_CUDA
noinst_LTLIBRARIES += libIniConfigGpu.la
endif

# libIniConfig parsed/compiled with g++
libIniConfig_la_SOURCES = \
	inih/ini.h \
	inih/ini.c \
	inih/INIReader.h \
	inih/INIReader.cpp \
	ConfigMap.h \
	ConfigMap.cpp
libIniConfig_la_CPPFLAGS = $(AM_CPPFLAGS)

if USE_CUDA
# libIniConfig parsed/compiled with nvcc
BUILT_SOURCES = inih/ini.cu inih/INIReader.cu ConfigMap.cu ConfigMapTest.cu
CLEANFILES    = inih/ini.cu inih/INIReader.cu ConfigMap.cu ConfigMapTest.cu
libIniConfigGpu_la_SOURCES = \
	inih/ini.h \
	inih/INIReader.h \
	ConfigMap.h 
nodist_libIniConfigGpu_la_SOURCES = \
	inih/ini.cu \
	inih/INIReader.cu \
	ConfigMap.cu
nodist_EXTRA_libIniConfigGpu_la_SOURCES = dummy.cpp
libIniConfigGpu_la_CPPFLAGS = $(AM_CPPFLAGS) $(CUDA_CFLAGS)
endif

# Some test programs
bin_PROGRAMS = INIReaderTest ConfigMapTest
if USE_CUDA
bin_PROGRAMS += ConfigMapTestGpu
endif 

INIReaderTest_SOURCES = inih/INIReaderTest.cpp
INIReaderTest_LDFLAGS = libIniConfig.la
INIReaderTest_DEPENDENCIES = libIniConfig.la

ConfigMapTest_SOURCES = ConfigMapTest.cpp
ConfigMapTest_CPPFLAGS =
ConfigMapTest_LDFLAGS = libIniConfig.la
ConfigMapTest_DEPENDENCIES = libIniConfig.la

if USE_CUDA
nodist_ConfigMapTestGpu_SOURCES = ConfigMapTest.cu
# add dummy C++ source to cause C++ linking.
nodist_EXTRA_ConfigMapTestGpu_SOURCES = dummy.cxx
ConfigMapTestGpu_CPPFLAGS = $(CUDA_CFLAGS) 
ConfigMapTestGpu_LDFLAGS  = $(CUDA_LIBS) libIniConfigGpu.la
ConfigMapTestGpu_DEPENDENCIES = libIniConfigGpu.la

endif
