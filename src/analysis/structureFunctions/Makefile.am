AUTOMAKE_OPTIONS = subdir-objects

# clean generated files
MAINTAINERCLEANFILES = $(srcdir)/Makefile.in

CXX=@MPICXX@
AM_CPPFLAGS = -Wno-unused-parameter -Wno-unused-function

# preprocessing flags
STRUCTURE_FUNCTIONS_C_FLAGS=
STRUCTURE_FUNCTIONS_LD_FLAGS=
if HAVE_PNETCDF
STRUCTURE_FUNCTIONS_C_FLAGS+= -DUSE_PNETCDF $(PNETCDF_CPPFLAGS)
STRUCTURE_FUNCTIONS_LD_FLAGS+=$(PNETCDF_LDFLAGS)
endif

############################################
# some analysis / post-processing programs
############################################
bin_PROGRAMS =

# power spectrum and other apps
bin_PROGRAMS += \
	computeStructureFunctionsMpi \
	computeStructureFunctionsMpi_v2 \
	computeStructureFunctionsMpi_v3 \
	test_pnetcdf_write


# structure functions
computeStructureFunctionsMpi_SOURCES = \
	pnetcdf_io.cpp \
	common_sf.cpp \
	structureFunctionsMpi.cpp

computeStructureFunctionsMpi_CPPFLAGS = $(AM_CPPFLAGS) $(STRUCTURE_FUNCTIONS_C_FLAGS) -I$(top_srcdir)/src/utils/config -I$(top_srcdir)/src/utils/cnpy -I$(top_srcdir)/src/hydro
computeStructureFunctionsMpi_LDFLAGS  = $(AM_LDFLAGS) ../../utils/config/libIniConfig.la $(STRUCTURE_FUNCTIONS_LD_FLAGS) ../../utils/cnpy/libCNpy.la


# structure functions (v2)
computeStructureFunctionsMpi_v2_SOURCES = \
	pnetcdf_io.cpp \
	common_sf.cpp \
	structureFunctionsMpi_v2.cpp

computeStructureFunctionsMpi_v2_CPPFLAGS = $(AM_CPPFLAGS) $(STRUCTURE_FUNCTIONS_C_FLAGS) -I$(top_srcdir)/src/utils/config -I$(top_srcdir)/src/utils/cnpy -I$(top_srcdir)/src/hydro
computeStructureFunctionsMpi_v2_LDFLAGS  = $(AM_LDFLAGS) ../../utils/config/libIniConfig.la $(STRUCTURE_FUNCTIONS_LD_FLAGS) ../../utils/cnpy/libCNpy.la


# structure functions (v3)
computeStructureFunctionsMpi_v3_SOURCES = \
	pnetcdf_io.cpp \
	common_sf.cpp \
	structureFunctionsMpi_v3.cpp

computeStructureFunctionsMpi_v3_CPPFLAGS = $(AM_CPPFLAGS) $(STRUCTURE_FUNCTIONS_C_FLAGS) -I$(top_srcdir)/src/utils/config -I$(top_srcdir)/src/utils/cnpy -I$(top_srcdir)/src/hydro
computeStructureFunctionsMpi_v3_LDFLAGS  = $(AM_LDFLAGS) ../../utils/config/libIniConfig.la $(STRUCTURE_FUNCTIONS_LD_FLAGS) ../../utils/cnpy/libCNpy.la


# test pnetcdf write routine
test_pnetcdf_write_SOURCES = \
	pnetcdf_io.cpp \
	test_pnetcdf_write.cpp

test_pnetcdf_write_CPPFLAGS = $(AM_CPPFLAGS) $(STRUCTURE_FUNCTIONS_C_FLAGS) -I$(top_srcdir)/src/utils/config -I$(top_srcdir)/src/hydro
test_pnetcdf_write_LDFLAGS  = $(AM_LDFLAGS) ../../utils/config/libIniConfig.la $(STRUCTURE_FUNCTIONS_LD_FLAGS) 

