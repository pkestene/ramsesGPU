AUTOMAKE_OPTIONS = subdir-objects

# clean generated files
MAINTAINERCLEANFILES = $(srcdir)/Makefile.in

TIMING_FLAGS =
if DO_TIMING
TIMING_FLAGS += -DDO_TIMING
endif

AM_CPPFLAGS = -Wno-unused-parameter -Wno-unused-function $(TIMING_FLAGS) -I$(top_srcdir)/src/utils/config

bin_PROGRAMS = testCopyBorderBuffer testCopyBorderBufferShear

testCopyBorderBuffer_SOURCES = \
	testCopyBorderBuffer.cpp
testCopyBorderBuffer_CPPFLAGS = $(AM_CPPFLAGS) -I$(srcdir)/../../src/hydro 
testCopyBorderBuffer_LDFLAGS  = $(AM_LDFLAGS) ../../src/hydro/libhydroCpu.la

testCopyBorderBufferShear_SOURCES = \
	testCopyBorderBufferShear.cpp
testCopyBorderBufferShear_CPPFLAGS = $(AM_CPPFLAGS) -I$(srcdir)/../../src/hydro 
testCopyBorderBufferShear_LDFLAGS  = $(AM_LDFLAGS) ../../src/hydro/libhydroCpu.la

##############################
# Some MPI test programs
##############################
if USE_MPI
bin_PROGRAMS += \
	testHydroMpiParameters \
	testMpiBuffer \
	testMpiOutputVtk \
	testHydroRunBaseMpi_border

##############################
testHydroMpiParameters_SOURCES = testHydroMpiParameters.cpp
testHydroMpiParameters_CPPFLAGS = $(AM_CPPFLAGS) $(MPI_CXXFLAGS) -I$(srcdir)/../../src/hydro -I$(srcdir)/../../src/utils/mpiUtils
testHydroMpiParameters_LDFLAGS = $(AM_LDFLAGS) $(MPI_LDFLAGS) ../../src/hydro/libhydroCpu.la ../../src/utils/config/libIniConfig.la ../../src/utils/mpiUtils/libMpiUtils.la 

##############################
testMpiBuffer_SOURCES = testMpiBuffer.cpp
testMpiBuffer_CPPFLAGS = $(AM_CPPFLAGS) $(MPI_CXXFLAGS) -I$(srcdir)/../../src/hydro -I$(srcdir)/../../src/utils/mpiUtils
testMpiBuffer_LDFLAGS = $(AM_LDFLAGS) $(MPI_LDFLAGS) ../../src/hydro/libhydroCpu.la ../../src/utils/mpiUtils/libMpiUtils.la 

##############################
testMpiOutputVtk_SOURCES = testMpiOutputVtk.cpp
testMpiOutputVtk_CPPFLAGS = $(AM_CPPFLAGS) $(MPI_CXXFLAGS) -I$(srcdir)/../../src/hydro -I$(srcdir)/../../src/utils/config -I$(srcdir)/../../src/utils/mpiUtils
testMpiOutputVtk_LDFLAGS = $(AM_LDFLAGS) $(MPI_LDFLAGS) ../../src/hydro/libhydroCpu.la ../../src/utils/config/libIniConfig.la ../../src/utils/mpiUtils/libMpiUtils.la ../../src/utils/cnpy/libCNpy.la 
if DO_TIMING
testMpiOutputVtk_LDFLAGS += ../../src/utils/monitoring/libMonitoring.la 
endif

##############################
testHydroRunBaseMpi_border_SOURCES = testHydroRunBaseMpi_border.cpp
testHydroRunBaseMpi_border_CPPFLAGS = $(AM_CPPFLAGS) $(MPI_CXXFLAGS) -I$(srcdir)/../../src/hydro -I$(srcdir)/../../src/utils/config -I$(srcdir)/../../src/utils/mpiUtils
testHydroRunBaseMpi_border_LDFLAGS = $(AM_LDFLAGS) $(MPI_LDFLAGS) ../../src/hydro/libhydroCpu.la ../../src/utils/config/libIniConfig.la ../../src/utils/mpiUtils/libMpiUtils.la ../../src/utils/cnpy/libCNpy.la 
if DO_TIMING
testHydroRunBaseMpi_border_LDFLAGS += ../../src/utils/monitoring/libMonitoring.la
endif

endif
