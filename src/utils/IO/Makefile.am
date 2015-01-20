AUTOMAKE_OPTIONS = subdir-objects

# clean generated files
MAINTAINERCLEANFILES = $(srcdir)/Makefile.in

# HDF5 flags
if HAVE_HDF5
HDF5_CPPFLAGS += -DUSE_HDF5 -DH5_NO_DEPRECATED_SYMBOLS 
endif

if HAVE_HDF5_PARALLEL
HDF5_CPPFLAGS += -DUSE_HDF5_PARALLEL
endif

# Parallel-netCDF flags
if HAVE_PNETCDF
PNETCDF_CPPFLAGS += -DUSE_PNETCDF
endif

# common flags
AM_CXXFLAGS = -I$(srcdir)/. $(HDF5_CPPFLAGS) $(PNETCDF_CPPFLAGS) $(MPI_CXXFLAGS) $(TIMING_FLAGS)
AM_CPPFLAGS = -I$(srcdir)/. $(HDF5_CPPFLAGS) $(PNETCDF_CPPFLAGS) $(MPI_CXXFLAGS) $(TIMING_FLAGS) 
# -Wno-unused-parameter -Wno-unused-function
AM_LDFLAGS  = $(HDF5_LDFLAGS) $(PNETCDF_LDFLAGS) $(MPI_LDFLAGS)

# default programs to build
bin_PROGRAMS =

# Some MPI programs
if USE_MPI
bin_PROGRAMS += \
	ioConvert

if USE_MPI_AND_DOUBLE_PRECISION
bin_PROGRAMS += ioConvert_double
endif

### SINGLE PRECISION ###
ioConvert_SOURCES  = ioConvert.cpp
ioConvert_CXXFLAGS = $(MPI_CXXFLAGS) -I$(top_srcdir)/src/utils/mpiUtils -I$(top_srcdir)/src $(EXTRA_FLAGS)
ioConvert_LDADD    = $(MPI_LDFLAGS) ../config/libIniConfig.la ../monitoring/libMonitoring.la ../mpiUtils/libMpiUtils.la ../cnpy/libCNpy.la

### DOUBLE PRECISION ###
ioConvert_double_SOURCES  = ioConvert.cpp
ioConvert_double_CXXFLAGS = -DUSE_DOUBLE $(MPI_CXXFLAGS) -I$(top_srcdir)/src/utils/mpiUtils -I$(top_srcdir)/src $(EXTRA_FLAGS)
#ioConvert_double_CPPFLAGS = -DUSE_DOUBLE $(AM_CPPFLAGS)
ioConvert_double_LDADD    = $(MPI_LDFLAGS) ../config/libIniConfig.la ../monitoring/libMonitoring.la ../mpiUtils/libMpiUtils.la ../cnpy/libCNpy.la
endif
