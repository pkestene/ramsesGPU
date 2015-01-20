AUTOMAKE_OPTIONS = subdir-objects

# clean generated files
MAINTAINERCLEANFILES = $(srcdir)/Makefile.in

SUBDIRS = . mpiBasic mpiHydro 

if USE_MPI
if USE_CUDA
SUBDIRS += mpiCuda
endif
endif

AM_CFLAGS = $(CUDA_CFLAGS)

bin_PROGRAMS = makeConfigHydro 
makeConfigHydro_SOURCES = makeConfigHydro.cpp
makeConfigHydro_CPPFLAGS = -I$(top_srcdir)/src/hydro
