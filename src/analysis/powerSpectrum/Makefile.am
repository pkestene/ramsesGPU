AUTOMAKE_OPTIONS = subdir-objects

# clean generated files
MAINTAINERCLEANFILES = $(srcdir)/Makefile.in

CXX=@MPICXX@
AM_CPPFLAGS = -Wno-unused-parameter -Wno-unused-function

# preprocessing flags
POWER_SPECTRUM_C_FLAGS=
POWER_SPECTRUM_LD_FLAGS=
if HAVE_PNETCDF
POWER_SPECTRUM_C_FLAGS+= -DUSE_PNETCDF $(PNETCDF_CPPFLAGS)
POWER_SPECTRUM_LD_FLAGS+=$(PNETCDF_LDFLAGS)
endif

if HAVE_FFTW3_MPI
POWER_SPECTRUM_C_FLAGS+= -DUSE_FFTW3_MPI $(FFTW3_CPPFLAGS)
POWER_SPECTRUM_LD_FLAGS+=$(FFTW3_LDFLAGS)
endif

############################################
# some analysis / post-processing programs
############################################
bin_PROGRAMS =

# power spectrum and other apps
bin_PROGRAMS += \
	computePowerSpectrumMpi \
	computeProjectedDensityMpi \
	generate_fBm


# power spectrum
computePowerSpectrumMpi_SOURCES = \
	fft_mpi.cpp \
	pnetcdf_io.cpp \
	powerSpectrumMpi.cpp

computePowerSpectrumMpi_CPPFLAGS = $(AM_CPPFLAGS) $(POWER_SPECTRUM_C_FLAGS) -I$(top_srcdir)/src/utils/config -I$(top_srcdir)/src/utils/cnpy -I$(top_srcdir)/src/hydro
computePowerSpectrumMpi_LDFLAGS  = $(AM_LDFLAGS) ../../utils/config/libIniConfig.la $(POWER_SPECTRUM_LD_FLAGS) ../../utils/cnpy/libCNpy.la

# projected density
computeProjectedDensityMpi_SOURCES = \
	pnetcdf_io.cpp \
	projectedDensityMpi.cpp

computeProjectedDensityMpi_CPPFLAGS = $(AM_CPPFLAGS) $(POWER_SPECTRUM_C_FLAGS) -I$(top_srcdir)/src/utils/config -I$(top_srcdir)/src/utils/cnpy -I$(top_srcdir)/src/hydro
computeProjectedDensityMpi_LDFLAGS  = $(AM_LDFLAGS) ../../utils/config/libIniConfig.la $(POWER_SPECTRUM_LD_FLAGS) ../../utils/cnpy/libCNpy.la

# generate fBm
generate_fBm_SOURCES = \
	fft_mpi.cpp \
	pnetcdf_io.cpp \
	generate_fBm.cpp

generate_fBm_CXXFLAGS = -std=c++11 -DGEN_FBM
generate_fBm_CPPFLAGS = $(AM_CPPFLAGS) $(POWER_SPECTRUM_C_FLAGS) -I$(top_srcdir)/src/utils/config -I$(top_srcdir)/src/utils/cnpy -I$(top_srcdir)/src/hydro -std=c++11 -DGEN_FBM
generate_fBm_LDFLAGS  = $(AM_LDFLAGS) ../../utils/config/libIniConfig.la $(POWER_SPECTRUM_LD_FLAGS) ../../utils/cnpy/libCNpy.la

