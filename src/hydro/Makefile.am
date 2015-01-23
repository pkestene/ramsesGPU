# take care that autotroll.am 
# already defines the following variables
# SUFFIXES, DISTCLEANFILES and CLEANFILES
include $(top_srcdir)/am/autotroll.am
# also note that SUFFIXES must be set with .cu before including cuda.am
# ortherwise this Makefile is not usable
SUFFIXES += .cu
include $(top_srcdir)/am/cuda.am

# clean generated files
MAINTAINERCLEANFILES = $(srcdir)/Makefile.in 

AUTOMAKE_OPTIONS = subdir-objects

# define some usefull preprocessor symbols
CONFIG_CPPFLAGS = -I$(top_srcdir)/src/utils/config
CNPY_CPPFLAGS   = -I$(top_srcdir)/src/utils/cnpy

#VTK_CPPFLAGS=
if HAVE_VTK
VTK_CPPFLAGS += -DUSE_VTK
endif

# HDF5 flags
if HAVE_HDF5
HDF5_CPPFLAGS += -DUSE_HDF5 -DH5_NO_DEPRECATED_SYMBOLS 
endif

if HAVE_HDF5_PARALLEL
HDF5_CPPFLAGS += -DUSE_HDF5_PARALLEL
endif

# NetCDF4 flags (for c++)
if HAVE_NETCDF4
NETCDF_CPPFLAGS += -DUSE_NETCDF4
#NETCDF_LDFLAGS += -lnetcdf_c++
endif

# Parallel-netCDF flags
if HAVE_PNETCDF
PNETCDF_CPPFLAGS += -DUSE_PNETCDF
endif

# GraphicsMagick++ cflags
GM_CPPFLAGS= $(GM_CFLAGS)
if HAVE_GM
GM_CPPFLAGS += -DUSE_GM
endif

# PAPI performance measurements
if HAVE_PAPI
PAPI_CPPFLAGS += -DUSE_PAPI
endif

# timing flags
TIMING_FLAGS =
if DO_TIMING
TIMING_FLAGS += -DDO_TIMING
endif

# define AM_* flags common to all targets
AM_CXXFLAGS = -I$(srcdir)/. $(CONFIG_CPPFLAGS) $(CNPY_CPPFLAGS) $(CUDA_CFLAGS) $(QT_CXXFLAGS) $(VTK_CPPFLAGS) $(HDF5_CPPFLAGS) $(NETCDF_CPPFLAGS) $(PNETCDF_CPPFLAGS) $(GM_CPPFLAGS) $(MPI_CXXFLAGS) $(TIMING_FLAGS) $(PAPI_CPPFLAGS)
AM_CFLAGS   = -I$(srcdir)/. $(CONFIG_CPPFLAGS) $(CNPY_CPPFLAGS) $(CUDA_CFLAGS) $(QT_CPPFLAGS) $(VTK_CPPFLAGS) $(HDF5_CPPFLAGS) $(NETCDF_CPPFLAGS) $(PNETCDF_CPPFLAGS) $(GM_CPPFLAGS) $(MPI_CXXFLAGS) $(TIMING_FLAGS) $(PAPI_CPPFLAGS)
AM_CPPFLAGS = -I$(srcdir)/. $(CONFIG_CPPFLAGS) $(CNPY_CPPFLAGS) $(CUDA_CFLAGS) $(QT_CPPFLAGS) $(VTK_CPPFLAGS) $(HDF5_CPPFLAGS) $(NETCDF_CPPFLAGS) $(PNETCDF_CPPFLAGS) $(GM_CPPFLAGS) $(MPI_CXXFLAGS) $(TIMING_FLAGS) $(PAPI_CPPFLAGS)
AM_LDFLAGS  = $(VTK_LDFLAGS) $(VTK_LIBS)  $(HDF5_LDFLAGS) $(NETCDF_LDFLAGS) $(PNETCDF_LDFLAGS) $(GM_CPPFLAGS) $(MPI_LDFLAGS) $(PAPI_LDFLAGS)
AM_FCFLAGS  = -fno-underscoring
#if USE_MPI
#AM_CXXFLAGS += -DUSE_MPI
#AM_CFLAGS   += -DUSE_MPI
#AM_CPPFLAGS += -DUSE_MPI
#endif

# the following makes sure that some of the macros defined by autoconf/automake
# are also passed to nvcc
NVCCFLAGS += $(filter -DHAVE%, $(DEFS))

NVCCFLAGS += --ptxas-options -v
NVCCFLAGS += --compiler-options -I$(srcdir) --compiler-options $(CONFIG_CPPFLAGS) 
NVCCFLAGS += --compiler-options -I$(srcdir)/glutGui
NVCCFLAGS += --compiler-options -I$(srcdir)/qtGui/qtHydro2d
NVCCFLAGS += $(addprefix --compiler-options ,$(QT_CPPFLAGS)) 
NVCCFLAGS += $(addprefix --compiler-options ,$(QT_CXXFLAGS))
NVCCFLAGS += --compiler-options -fno-strict-aliasing
NVCCFLAGS += $(addprefix --compiler-options ,$(VTK_CPPFLAGS) )
NVCCFLAGS += $(addprefix --compiler-options ,$(HDF5_CPPFLAGS) )
NVCCFLAGS += $(addprefix --compiler-options ,$(NETCDF_CPPFLAGS) )
NVCCFLAGS += $(addprefix --compiler-options ,$(PNETCDF_CPPFLAGS) )
if HAVE_GM
NVCCFLAGS += $(addprefix --compiler-options ,$(GM_CPPFLAGS) )
endif
NVCCFLAGS += $(addprefix --compiler-options ,$(MPI_CXXFLAGS) )
NVCCFLAGS += --compiler-options -I$(srcdir)/../utils/mpiUtils
NVCCFLAGS += --compiler-options $(CNPY_CPPFLAGS) 
NVCCFLAGS += --compiler-options -Wno-unused-function
NVCCFLAGS += --compiler-options -Wno-unused-parameter
if DO_TIMING
NVCCFLAGS += --compiler-options -DDO_TIMING
endif

# -deviceemu
#-deviceemu -keep


# ===================================================
# Some SOURCES list variables
# ===================================================
# common sources code for both sequential (CPU+CUDA) and parallel (MPI+CUDA)
HYDRO_SRC = \
	real_type.h \
	base_type.h \
	ostream_fmt.h \
	common_types.h \
	utilities.h \
	gpu_macros.h \
	gpu_macros.cpp \
	make_boundary_base.h \
	make_boundary_common.h \
	make_boundary_shear.h \
	Arrays.h \
	GetPot.h \
	constants.h \
	constants.cpp \
	constoprim.h \
	cmpflx.h \
	riemann.h \
	riemann_mhd.h \
	mhd_utils.h \
	trace.h \
	trace_mhd.h \
	slope.h \
	slope_mhd.h \
	turbulenceInit.h \
	turbulenceInit.cpp \
	structureFunctions.h \
	structureFunctions.cpp \
	kurganov-tadmor.h \
	positiveScheme.h \
	initHydro.h \
	initHydro.cpp \
	vtk_inc.h \
	RandomGen.h \
	RandomGen.cpp \
	Forcing_OrnsteinUhlenbeck.h \
	Forcing_OrnsteinUhlenbeck.cpp \
	HydroParameters.h \
	HydroRunBase.h \
	HydroRunBase.cpp \
	HydroRunGodunov.h \
	HydroRunGodunov.cpp \
	HydroRunKT.h \
	HydroRunKT.cpp \
	relaxingTVD.h \
	HydroRunRelaxingTVD.h \
	HydroRunRelaxingTVD.cpp \
	MHDRunBase.h \
	MHDRunBase.cpp \
	MHDRunGodunov.h \
	MHDRunGodunov.cpp \
	shearBorderUtils.h \
	HydroRunGodunovZslab.h \
	HydroRunGodunovZslab.cpp \
	MHDRunGodunovZslab.h \
	MHDRunGodunovZslab.cpp
if USE_MPI
HYDRO_SRC += \
	mpiBorderUtils.h \
	HydroMpiParameters.h \
	HydroMpiParameters.cpp \
	HydroRunBaseMpi.h \
	HydroRunBaseMpi.cpp \
	HydroRunGodunovMpi.h \
	HydroRunGodunovMpi.cpp \
	MHDRunGodunovMpi.h \
	MHDRunGodunovMpi.cpp \
	HydroRunGodunovZslabMpi.h \
	HydroRunGodunovZslabMpi.cpp \
	MHDRunGodunovZslabMpi.h \
	MHDRunGodunovZslabMpi.cpp \
	structureFunctionsMpi.h \
	structureFunctionsMpi.cpp 
endif

HYDRO_SRC_CUDA = \
	real_type.h \
	base_type.h \
	ostream_fmt.h \
	common_types.h \
	utilities.h \
	cutil.h cutil_inline.h cutil_inline_runtime.h \
	gpu_macros.h \
	make_boundary_base.h \
	make_boundary_common.h \
	make_boundary_shear.h \
	Arrays.h \
	GetPot.h \
	constants.h \
	constoprim.h \
	cmpflx.h \
	riemann.h \
	riemann_mhd.h \
	mhd_utils.h \
	trace.h \
	trace_mhd.h \
	slope.h \
	slope_mhd.h \
	turbulenceInit.h \
	turbulenceInit.cpp \
	structureFunctions.h \
	godunov.cuh \
	godunov_notrace.cuh \
	godunov_trace_v1.cuh \
	godunov_trace_v2.cuh \
	godunov_unsplit.cuh \
	godunov_unsplit_mhd.cuh \
	cmpdt.cuh \
	cmpdt_mhd.cuh \
	hydro_update.cuh \
	random_forcing.cuh \
	RandomGen.h \
	Forcing_OrnsteinUhlenbeck.h \
	Forcing_OrnsteinUhlenbeck_kernels.cuh \
	mhd_ct_update.cuh \
	kurganov-tadmor.h \
	kurganov-tadmor.cuh \
	laxliu.cuh \
	relaxingTVD.cuh \
	viscosity.cuh \
	viscosity_zslab.cuh \
	resistivity.cuh \
	resistivity_zslab.cuh \
	gravity.cuh \
	copyFaces.cuh \
	shearingBox_utils.cuh \
	shearingBox_utils_zslab.cuh \
	initHydro.h \
	initHydro.cpp \
	HydroParameters.h \
	HydroRunBase.h \
	HydroRunGodunov.h \
	HydroRunKT.h \
	HydroRunRelaxingTVD.h 

HYDRO_SRC_CUDA_SIMPLE = $(HYDRO_SRC_CUDA) \
	gpu_macros.cu \
	RandomGen.cu \
	Forcing_OrnsteinUhlenbeck.cu \
	HydroRunBase.cu \
	HydroRunGodunov.cu \
	HydroRunGodunovZslab.cu \
	HydroRunKT.cu \
	HydroRunRelaxingTVD.cu \
	MHDRunBase.cu \
	MHDRunGodunov.cu \
	MHDRunGodunovZslab.cu \
	structureFunctions.cu

HYDRO_SRC_CUDA_DOUBLE = $(HYDRO_SRC_CUDA) \
	gpu_macros-double.cu \
	RandomGen-double.cu \
	Forcing_OrnsteinUhlenbeck-double.cu \
	HydroRunBase-double.cu \
	HydroRunGodunov-double.cu \
	HydroRunGodunovZslab-double.cu \
	HydroRunKT-double.cu \
	HydroRunRelaxingTVD-double.cu \
	MHDRunBase-double.cu \
	MHDRunGodunov-double.cu \
	MHDRunGodunovZslab-double.cu \
	structureFunctions-double.cu


if USE_MPI
HYDRO_SRC_CUDA_SIMPLE += \
	mpiBorderUtils.cuh \
	HydroMpiParameters.h \
	HydroMpiParameters.cu \
	HydroRunBaseMpi.h \
	HydroRunBaseMpi.cu \
	HydroRunGodunovMpi.h \
	HydroRunGodunovMpi.cu \
	MHDRunGodunovMpi.h \
	MHDRunGodunovMpi.cu \
	HydroRunGodunovZslabMpi.h \
	HydroRunGodunovZslabMpi.cu \
	MHDRunGodunovZslabMpi.h \
	MHDRunGodunovZslabMpi.cu \
	structureFunctionsMpi.h \
	structureFunctionsMpi.cu
endif

if USE_MPI_AND_DOUBLE_PRECISION
HYDRO_SRC_CUDA_DOUBLE += \
	HydroMpiParameters.h \
	HydroMpiParameters-double.cu \
	HydroRunBaseMpi.h \
	HydroRunBaseMpi-double.cu \
	HydroRunGodunovMpi.h \
	HydroRunGodunovMpi-double.cu \
	MHDRunGodunovMpi.h \
	MHDRunGodunovMpi-double.cu \
	HydroRunGodunovZslabMpi.h \
	HydroRunGodunovZslabMpi-double.cu \
	MHDRunGodunovZslabMpi.h \
	MHDRunGodunovZslabMpi-double.cu \
	structureFunctionsMpi.h \
	structureFunctionsMpi-double.cu 
endif

# ===================================================
# Some convenience libraries
# ===================================================

#
# sequential (mono CPU) or MPI (multi-CPU)
#

# simple precision
noinst_LTLIBRARIES = libhydroCpu.la
libhydroCpu_la_SOURCES = $(HYDRO_SRC)
libhydroCpu_la_CPPFLAGS = $(AM_CPPFLAGS) -I$(srcdir)/../utils/mpiUtils
libhydroCpu_la_CXXFLAGS = $(AM_CPPFLAGS) -I$(srcdir)/../utils/mpiUtils -std=c++11

# double precision
if USE_DOUBLE_PRECISION
noinst_LTLIBRARIES += libhydroCpu_double.la
libhydroCpu_double_la_SOURCES = $(HYDRO_SRC)
libhydroCpu_double_la_CPPFLAGS = $(AM_CPPFLAGS) -DUSE_DOUBLE -I$(srcdir)/../utils/mpiUtils
libhydroCpu_double_la_CXXFLAGS = $(AM_CPPFLAGS) -DUSE_DOUBLE -I$(srcdir)/../utils/mpiUtils -std=c++11
endif

#
# CUDA sequential (mono CPU - mono GPU) or CUDA - MPI (multi-GPU)
#
if USE_CUDA

### SIMPLE PRECISION ###
BUILT_SOURCES = gpu_macros.cu RandomGen.cu Forcing_OrnsteinUhlenbeck.cu HydroRunBase.cu HydroRunGodunov.cu HydroRunGodunovZslab.cu HydroRunKT.cu HydroRunRelaxingTVD.cu HydroRunLaxLiu.cu MHDRunBase.cu MHDRunGodunov.cu MHDRunGodunovZslab.cu structureFunctions.cu
CLEANFILES   += gpu_macros.cu RandomGen.cu Forcing_OrnsteinUhlenbeck.cu HydroRunBase.cu HydroRunGodunov.cu HydroRunGodunovZslab.cu HydroRunKT.cu HydroRunRelaxingTVD.cu HydroRunLaxLiu.cu MHDRunBase.cu MHDRunGodunov.cu MHDRunGodunovZslab.cu structureFunctions.cu 

### DOUBLE PRECISION ###
BUILT_SOURCES += gpu_macros-double.cu RandomGen-double.cu Forcing_OrnsteinUhlenbeck-double.cu HydroRunBase-double.cu HydroRunGodunov-double.cu HydroRunGodunovZslab-double.cu HydroRunKT-double.cu HydroRunRelaxingTVD-double.cu MHDRunBase-double.cu MHDRunGodunov-double.cu MHDRunGodunovZslab-double.cu structureFunctions-double.cu
CLEANFILES    += gpu_macros-double.cu RandomGen-double.cu Forcing_OrnsteinUhlenbeck-double.cu HydroRunBase-double.cu HydroRunGodunov-double.cu HydroRunGodunovZslab-double.cu HydroRunKT-double.cu HydroRunRelaxingTVD-double.cu MHDRunBase-double.cu MHDRunGodunov-double.cu MHDRunGodunovZslab-double.cu structureFunctions-double.cu

if USE_MPI
BUILT_SOURCES += HydroMpiParameters.cu HydroRunBaseMpi.cu HydroRunGodunovMpi.cu MHDRunGodunovMpi.cu HydroRunGodunovZslabMpi.cu MHDRunGodunovZslabMpi.cu structureFunctionsMpi.cu
CLEANFILES    += HydroMpiParameters.cu HydroRunBaseMpi.cu HydroRunGodunovMpi.cu MHDRunGodunovMpi.cu HydroRunGodunovZslabMpi.cu MHDRunGodunovZslabMpi.cu structureFunctionsMpi.cu
endif

if USE_MPI_AND_DOUBLE_PRECISION
BUILT_SOURCES += HydroMpiParameters-double.cu HydroRunBaseMpi-double.cu HydroRunGodunovMpi-double.cu MHDRunGodunovMpi-double.cu HydroRunGodunovZslabMpi-double.cu MHDRunGodunovZslabMpi-double.cu structureFunctions-double.cu
CLEANFILES    += HydroMpiParameters-double.cu HydroRunBaseMpi-double.cu HydroRunGodunovMpi-double.cu MHDRunGodunovMpi-double.cu HydroRunGodunovZslabMpi-double.cu MHDRunGodunovZslabMpi-double.cu structureFunctions-double.cu
endif

# simple precision 
noinst_LTLIBRARIES += libhydroGpu.la
libhydroGpu_la_SOURCES = $(HYDRO_SRC_CUDA_SIMPLE)
libhydroGpu_la_CPPFLAGS = $(AM_CPPFLAGS) $(CUDA_CFLAGS) -I$(srcdir)/../utils/mpiUtils
libhydroGpu_la_CXXFLAGS = $(AM_CXXFLAGS) $(CUDA_CFLAGS) -I$(srcdir)/../utils/mpiUtils  -std=c++11

# double precision 
if USE_DOUBLE_PRECISION
noinst_LTLIBRARIES += libhydroGpu_double.la
libhydroGpu_double_la_SOURCES = $(HYDRO_SRC_CUDA_DOUBLE)
libhydroGpu_double_la_CPPFLAGS = $(AM_CPPFLAGS) $(CUDA_CFLAGS) -I$(srcdir)/../utils/mpiUtils -DUSE_DOUBLE
libhydroGpu_double_la_CXXFLAGS = $(AM_CXXFLAGS) $(CUDA_CFLAGS) -I$(srcdir)/../utils/mpiUtils -DUSE_DOUBLE -std=c++11
endif

endif
