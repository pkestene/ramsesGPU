AUTOMAKE_OPTIONS = subdir-objects

# clean generated files
MAINTAINERCLEANFILES = $(srcdir)/Makefile.in

# Parallel-netCDF flags
if HAVE_PNETCDF
PNETCDF_CPPFLAGS += -DUSE_PNETCDF
endif

# common flags
AM_CXXFLAGS = -I$(srcdir)/. $(PNETCDF_CPPFLAGS) $(MPI_CXXFLAGS) $(TIMING_FLAGS)
AM_CPPFLAGS = -I$(srcdir)/. $(PNETCDF_CPPFLAGS) $(MPI_CXXFLAGS) $(TIMING_FLAGS) -Wno-unused-parameter -Wno-unused-function
AM_LDFLAGS  = $(PNETCDF_LDFLAGS) $(MPI_LDFLAGS)

############################################
# some program
############################################
bin_PROGRAMS =

# read slice (for large data)
if USE_MPI
bin_PROGRAMS += \
	readSlice

if USE_MPI_AND_DOUBLE_PRECISION
bin_PROGRAMS += readSlice_double
endif

### SINGLE PRECISION ###
readSlice_SOURCES  = readSlice.cpp
readSlice_CXXFLAGS = $(MPI_CXXFLAGS) -I$(top_srcdir)/src/utils/mpiUtils -I$(top_srcdir)/src/utils/config -I$(top_srcdir)/src/utils/cnpy -I$(top_srcdir)/src
readSlice_LDADD    = $(MPI_LDFLAGS) ../../utils/config/libIniConfig.la ../../utils/cnpy/libCNpy.la ../../utils/mpiUtils/libMpiUtils.la

### DOUBLE PRECISION ###
readSlice_double_SOURCES  = readSlice.cpp
readSlice_double_CXXFLAGS = -DUSE_DOUBLE $(MPI_CXXFLAGS) -I$(top_srcdir)/src/utils/mpiUtils -I$(top_srcdir)/src/utils/config -I$(top_srcdir)/src/utils/cnpy -I$(top_srcdir)/src
readSlice_double_LDADD    = $(MPI_LDFLAGS) ../../utils/config/libIniConfig.la ../../utils/cnpy/libCNpy.la ../../utils/mpiUtils/libMpiUtils.la

endif
