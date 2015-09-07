# take care that autotroll.am 
# already defines the following variables
# SUFFIXES, DISTCLEANFILES and CLEANFILES
include $(top_srcdir)/am/autotroll.am
# also note that SUFFIXES must be set with .cu before including cuda.am
# ortherwise this Makefile is not usable
SUFFIXES += .cu
include $(top_srcdir)/am/cuda.am

######################################################
# svn version: force creation of file svn_version.cpp
######################################################
# BUILT_SOURCES=svn_version.cpp svn_version.h build_date.cpp build_date.h svn_dist_version
# EXTRA_DIST=svn_dist_version
# svn_version.cpp: FORCE
# 	echo -n 'const char* svn_version(void) { const char* SVN_Version = "' > svn_version.cpp
# 	@MYSVNVERSION@ >> svn_version.cpp
# 	echo '"; return SVN_Version; }' >> svn_version.cpp

# svn_version.h: FORCE
# 	echo -n 'const char* svn_version(void);' > svn_version.h

# build_date.cpp: FORCE
# 	echo -n 'const char* build_date(void) { const char* build_date_str = "' > build_date.cpp
# 	echo -n `date '+%Y-%m-%d %H:%M:%S %Z'` >> build_date.cpp
# 	echo '"; return build_date_str; };' >> build_date.cpp

# build_date.h: FORCE
# 	echo -n 'const char* build_date(void);' > build_date.h

BUILT_SOURCES=
EXTRA_DIST=

if HAVE_SVN_WC
#
# we are building from a svn source tree or a release source tree (from make dist)
svn_dist_version: FORCE
	@MYSVNVERSION@ > svn_dist_version
else
# the following rule is only called if file svn_dist_version
# doesn't exist, which should only happen when build a source tree
# obtained from svn export (and not a make dist)
#
svn_dist_version:
	echo -n 'Unknow version (sources probably obtained from svn export)' > svn_dist_version

endif


FORCE:




####################################################
# clean generated files
MAINTAINERCLEANFILES = $(srcdir)/Makefile.in 

AUTOMAKE_OPTIONS = subdir-objects
SUBDIRS = hydro utils .
# if ENABLE_QTGUI
# SUBDIRS += ./qtGui/qtHelloWorld	
# endif

#VTK_CPPFLAGS=
if HAVE_VTK
VTK_CPPFLAGS += -DUSE_VTK
endif

TIMING_FLAGS =
if DO_TIMING
TIMING_FLAGS += -DDO_TIMING

# PAPI performance measurements
if HAVE_PAPI
PAPI_CPPFLAGS += -DUSE_PAPI
TIMING_FLAGS += -DUSE_PAPI
endif

endif


EXTRA_FLAGS =
if USE_MIXED_PRECISION
EXTRA_FLAGS += -DUSE_MIXED_PRECISION
endif

AM_CXXFLAGS = -I$(srcdir)/hydro -I$(srcdir)/utils/config -I$(srcdir)/utils/monitoring $(CUDA_CFLAGS) $(VTK_CPPFLAGS) $(TIMING_FLAGS)
AM_CFLAGS   = $(CUDA_CFLAGS) $(VTK_CPPFLAGS) $(TIMING_FLAGS)
AM_CPPFLAGS = -I$(srcdir)/hydro -I$(srcdir)/utils/config -I$(srcdir)/utils/monitoring $(CUDA_CFLAGS) $(VTK_CPPFLAGS) $(TIMING_FLAGS)
AM_LDFLAGS  = $(VTK_LDFLAGS) $(VTK_LIBS) 
AM_FCFLAGS  = -fno-underscoring
#NVCCFLAGS  += --host-compilation=c++
NVCCFLAGS  += --ptxas-options -v
NVCCFLAGS  += --compiler-options -I$(srcdir) \
	--compiler-options -I$(srcdir)/hydro \
	--compiler-options -I$(srcdir)/utils/config \
	--compiler-options -I$(srcdir)/utils/monitoring \
	--compiler-options -I$(srcdir)/glutGui 
NVCCFLAGS += --compiler-options -fno-strict-aliasing
NVCCFLAGS += $(addprefix --compiler-options ,$(VTK_CPPFLAGS) )
NVCCFLAGS += --compiler-options -Wno-unused-function
NVCCFLAGS += --compiler-options -Wno-unused-parameter
if DO_TIMING
NVCCFLAGS += --compiler-options -DDO_TIMING
endif

if USE_MPI
NVCCFLAGS  += $(addprefix --compiler-options , $(MPI_CXXFLAGS) )
NVCCFLAGS  += --compiler-options -I$(top_srcdir)/src/utils/mpiUtils
endif

if USE_MIXED_PRECISION
NVCCFLAGS += --compiler-options $(EXTRA_FLAGS)
endif

# -deviceemu
#-deviceemu -keep

# ===================================================
# PROGRAMS :
# - euler_cpu     (CPU serial   - NO MPI)
# - euler_cpu_mpi (CPU parallel -    MPI)
# - euler_gpu     (GPU serial   - NO MPI - CUDA)
# - euler_gpu_mpi (GPU parallel -    MPI - CUDA)
# ===================================================
bin_PROGRAMS  = euler_cpu \
		euler2d_laxliu_cpu \
		euler_zslab_cpu
if USE_DOUBLE_PRECISION
bin_PROGRAMS += euler_cpu_double \
		euler_zslab_cpu_double 
endif

if USE_MPI
bin_PROGRAMS += euler_cpu_mpi \
		euler_zslab_cpu_mpi
endif
if USE_MPI_AND_DOUBLE_PRECISION
bin_PROGRAMS += euler_cpu_mpi_double \
		euler_zslab_cpu_mpi_double
endif

if USE_CUDA
bin_PROGRAMS += euler_gpu \
		euler2d_laxliu_gpu \
		euler_zslab_gpu
if USE_DOUBLE_PRECISION
bin_PROGRAMS += euler_gpu_double \
		euler_zslab_gpu_double
endif

if USE_MPI
bin_PROGRAMS += euler_gpu_mpi \
		euler_zslab_gpu_mpi
endif
if USE_MPI_AND_DOUBLE_PRECISION
bin_PROGRAMS += euler_gpu_mpi_double \
		euler_zslab_gpu_mpi_double
endif
endif

GM_FLAGS_SUPP=
if HAVE_GM
GM_FLAGS_SUPP += -DUSE_GM
NVCCFLAGS += $(addprefix --compiler-options , -DUSE_GM $(GM_CFLAGS))
endif


# =======================================================
#
# Euler 2d/3d on CPU (Godunov-Riemann or Kurganov-Tadmor)
#
# =======================================================
### SIMPLE PRECISION ###
euler_cpu_SOURCES = euler_main.cpp
euler_cpu_CXXFLAGS = $(GM_CFLAGS) $(GM_FLAGS_SUPP) $(EXTRA_FLAGS)
euler_cpu_LDADD = $(GM_LIBS) hydro/libhydroCpu.la utils/config/libIniConfig.la utils/monitoring/libMonitoring.la utils/cnpy/libCNpy.la $(PAPI_LDFLAGS)

### DOUBLE PRECISION ###
if USE_DOUBLE_PRECISION
euler_cpu_double_SOURCES = euler_main.cpp
euler_cpu_double_CXXFLAGS = -DUSE_DOUBLE $(GM_CFLAGS) $(GM_FLAGS_SUPP)
euler_cpu_double_CPPFLAGS = -DUSE_DOUBLE $(AM_CPPFLAGS) 
euler_cpu_double_LDADD = $(GM_LIBS) hydro/libhydroCpu_double.la utils/config/libIniConfig.la utils/monitoring/libMonitoring.la utils/cnpy/libCNpy.la $(PAPI_LDFLAGS)
endif

# ============================
#
# Euler 2d/3d on CPU with MPI
#
# ============================
### Simple Precision - MPI ###
if USE_MPI
euler_cpu_mpi_SOURCES = euler_mpi_main.cpp
euler_cpu_mpi_CXXFLAGS = $(MPI_CXXFLAGS) $(GM_CFLAGS) $(GM_FLAGS_SUPP) -I$(srcdir)/utils/mpiUtils  $(EXTRA_FLAGS)
euler_cpu_mpi_LDFLAGS = $(MPI_LDFLAGS) $(GM_LIBS)
euler_cpu_mpi_LDADD = hydro/libhydroCpu.la utils/config/libIniConfig.la utils/monitoring/libMonitoring.la utils/cnpy/libCNpy.la utils/mpiUtils/libMpiUtils.la
endif

### Double Precision - MPI ###
if USE_MPI_AND_DOUBLE_PRECISION
euler_cpu_mpi_double_SOURCES = euler_mpi_main.cpp
euler_cpu_mpi_double_CXXFLAGS = -DUSE_DOUBLE $(MPI_CXXFLAGS) $(GM_CFLAGS) $(GM_FLAGS_SUPP) -I$(srcdir)/utils/mpiUtils  $(EXTRA_FLAGS)
euler_cpu_mpi_double_CPPFLAGS = -DUSE_DOUBLE $(AM_CPPFLAGS)
euler_cpu_mpi_double_LDFLAGS = $(MPI_LDFLAGS) $(GM_LIBS)
euler_cpu_mpi_double_LDADD = hydro/libhydroCpu_double.la utils/config/libIniConfig.la utils/monitoring/libMonitoring.la utils/cnpy/libCNpy.la utils/mpiUtils/libMpiUtils.la
endif

# =======================================================
#
# Euler 2d/3d on GPU (Godunov-Riemann or Kurganov-Tadmor)
#
# =======================================================
if USE_CUDA

### SIMPLE PRECISION ###
BUILT_SOURCES += euler_main.cu
CLEANFILES    += euler_main.cu
nodist_euler_gpu_SOURCES = euler_main.cu 
# force CXX link
nodist_EXTRA_euler_gpu_SOURCES = dummy.cpp
euler_gpu_CPPFLAGS = $(CUDA_CFLAGS) $(GM_CFLAGS) $(GM_FLAGS_SUPP)
euler_gpu_LDFLAGS  = $(CUDA_LIBS) hydro/libhydroGpu.la utils/config/libIniConfigGpu.la utils/monitoring/libMonitoringGpu.la
euler_gpu_LDADD = $(GM_LIBS) utils/cnpy/libCNpy.la

### DOUBLE PRECISION ###
if USE_DOUBLE_PRECISION
BUILT_SOURCES += euler_main-double.cu
CLEANFILES    += euler_main-double.cu
nodist_euler_gpu_double_SOURCES = euler_main-double.cu 
# force CXX link
nodist_EXTRA_euler_gpu_double_SOURCES = dummy.cpp
euler_gpu_double_CPPFLAGS = -DUSE_DOUBLE $(CUDA_CFLAGS) $(GM_CFLAGS) $(GM_FLAGS_SUPP)
euler_gpu_double_CXXFLAGS = -DUSE_DOUBLE $(AM_CPPFLAGS) 
euler_gpu_double_LDFLAGS  = $(CUDA_LIBS) hydro/libhydroGpu_double.la utils/config/libIniConfigGpu.la utils/monitoring/libMonitoringGpu.la
euler_gpu_double_LDADD = $(GM_LIBS) utils/cnpy/libCNpy.la
endif

# =================================
#
# Euler 2d/3d on GPU with MPI+CUDA
#
# =================================
### SIMPLE PRECISION ###
if USE_MPI
BUILT_SOURCES += euler_mpi_main.cu
CLEANFILES    += euler_mpi_main.cu
nodist_euler_gpu_mpi_SOURCES = euler_mpi_main.cu
# force CXX link
nodist_EXTRA_euler_gpu_mpi_SOURCES = dummy.cpp
euler_gpu_mpi_CPPFLAGS = $(CUDA_CFLAGS) $(MPI_CXXFLAGS) $(GM_CFLAGS) $(GM_FLAGS_SUPP) -I$(srcdir)/utils/mpiUtils $(EXTRA_FLAGS)
euler_gpu_mpi_CXXFLAGS = $(CUDA_CFLAGS) $(MPI_CXXFLAGS) $(GM_CFLAGS) $(GM_FLAGS_SUPP) -I$(srcdir)/utils/mpiUtils $(EXTRA_FLAGS)
euler_gpu_mpi_LDFLAGS  = $(CUDA_LIBS) $(GM_LIBS) hydro/libhydroGpu.la utils/config/libIniConfigGpu.la utils/monitoring/libMonitoringGpu.la utils/cnpy/libCNpy.la utils/mpiUtils/libMpiUtils.la $(MPI_LDFLAGS) 
endif

### DOUBLE PRECISION ###
if USE_MPI_AND_DOUBLE_PRECISION
BUILT_SOURCES += euler_mpi_main-double.cu
CLEANFILES    += euler_mpi_main-double.cu
nodist_euler_gpu_mpi_double_SOURCES = euler_mpi_main-double.cu 
# force CXX link
nodist_EXTRA_euler_gpu_mpi_double_SOURCES = dummy.cpp
euler_gpu_mpi_double_CXXFLAGS = -DUSE_DOUBLE $(CUDA_CFLAGS) $(MPI_CXXFLAGS) $(GM_CFLAGS) $(GM_FLAGS_SUPP) -I$(srcdir)/utils/mpiUtils $(EXTRA_FLAGS)
euler_gpu_mpi_double_CPPFLAGS = -DUSE_DOUBLE $(AM_CPPFLAGS)
euler_gpu_mpi_double_LDFLAGS  = $(CUDA_LIBS)  $(GM_LIBS) hydro/libhydroGpu_double.la utils/config/libIniConfigGpu.la utils/monitoring/libMonitoringGpu.la utils/cnpy/libCNpy.la utils/mpiUtils/libMpiUtils.la $(MPI_LDFLAGS)
endif

endif


# =======================================================
#
# Euler 2d on CPU (Lax-Liu)
#
# =======================================================
euler2d_laxliu_cpu_SOURCES = \
	hydro/HydroRunLaxLiu.h \
	hydro/HydroRunLaxLiu.cpp \
	euler2d_laxliu.cpp
euler2d_laxliu_cpu_CXXFLAGS = $(GM_CFLAGS) $(GM_FLAGS_SUPP)
euler2d_laxliu_cpu_LDADD = $(GM_LIBS) -lm hydro/libhydroCpu.la utils/config/libIniConfig.la utils/monitoring/libMonitoring.la utils/cnpy/libCNpy.la

# =======================================================
#
# Euler 2d on GPU (Lax-Liu)
#
# =======================================================
if USE_CUDA
BUILT_SOURCES += euler2d_laxliu.cu
CLEANFILES    += euler2d_laxliu.cu
nodist_euler2d_laxliu_gpu_SOURCES = \
	hydro/HydroRunLaxLiu.h \
	hydro/HydroRunLaxLiu.cu \
	euler2d_laxliu.cu
# add dummy C++ source to cause C++ linking.
nodist_EXTRA_euler2d_laxliu_gpu_SOURCES = dummy.cxx
euler2d_laxliu_gpu_CPPFLAGS = $(CUDA_CFLAGS) $(GM_CFLAGS) $(GM_FLAGS_SUPP)
euler2d_laxliu_gpu_LDFLAGS  = $(CUDA_LIBS) 
euler2d_laxliu_gpu_LDADD    = hydro/libhydroGpu.la utils/config/libIniConfigGpu.la utils/monitoring/libMonitoringGpu.la utils/cnpy/libCNpy.la $(GM_LIBS)
endif


# =======================================================
#
# Euler 3d on CPU (Godunov - Z-SLAB)
#
# =======================================================
### SIMPLE PRECISION ###
euler_zslab_cpu_SOURCES = euler_zslab_main.cpp 
euler_zslab_cpu_CXXFLAGS = $(GM_CFLAGS) $(GM_FLAGS_SUPP) $(EXTRA_FLAGS)
euler_zslab_cpu_LDADD = $(GM_LIBS) hydro/libhydroCpu.la utils/config/libIniConfig.la utils/monitoring/libMonitoring.la utils/cnpy/libCNpy.la

### DOUBLE PRECISION ###
if USE_DOUBLE_PRECISION
euler_zslab_cpu_double_SOURCES = euler_zslab_main.cpp
euler_zslab_cpu_double_CXXFLAGS = -DUSE_DOUBLE $(GM_CFLAGS) $(GM_FLAGS_SUPP)
euler_zslab_cpu_double_CPPFLAGS = -DUSE_DOUBLE $(AM_CPPFLAGS) 
euler_zslab_cpu_double_LDADD = $(GM_LIBS) hydro/libhydroCpu_double.la utils/config/libIniConfig.la utils/monitoring/libMonitoring.la utils/cnpy/libCNpy.la
endif

# =======================================================
#
# Euler 3d on GPU (Godunov - Z-SLAB)
#
# =======================================================
if USE_CUDA

### SIMPLE PRECISION ###
BUILT_SOURCES += euler_zslab_main.cu
CLEANFILES    += euler_zslab_main.cu
nodist_euler_zslab_gpu_SOURCES = euler_zslab_main.cu
# force CXX link
nodist_EXTRA_euler_zslab_gpu_SOURCES = dummy.cpp
euler_zslab_gpu_CPPFLAGS = $(CUDA_CFLAGS) $(GM_CFLAGS) $(GM_FLAGS_SUPP)
euler_zslab_gpu_LDFLAGS  = $(CUDA_LIBS) hydro/libhydroGpu.la utils/config/libIniConfigGpu.la utils/monitoring/libMonitoringGpu.la utils/cnpy/libCNpy.la
euler_zslab_gpu_LDADD = $(GM_LIBS) 

### DOUBLE PRECISION ###
if USE_DOUBLE_PRECISION
BUILT_SOURCES += euler_zslab_main-double.cu
CLEANFILES    += euler_zslab_main-double.cu
nodist_euler_zslab_gpu_double_SOURCES = \
	euler_zslab_main-double.cu
# force CXX link
nodist_EXTRA_euler_zslab_gpu_double_SOURCES = dummy.cpp
euler_zslab_gpu_double_CPPFLAGS = -DUSE_DOUBLE $(CUDA_CFLAGS) $(GM_CFLAGS) $(GM_FLAGS_SUPP)
euler_zslab_gpu_double_CXXFLAGS = -DUSE_DOUBLE $(AM_CPPFLAGS) 
euler_zslab_gpu_double_LDFLAGS  = $(CUDA_LIBS) hydro/libhydroGpu_double.la utils/config/libIniConfigGpu.la utils/monitoring/libMonitoringGpu.la utils/cnpy/libCNpy.la
euler_zslab_gpu_double_LDADD = $(GM_LIBS)
endif

endif

# =================================
#
# Euler 3D on CPU with MPI - ZSLAB
#
# =================================
### Simple Precision - MPI ###
if USE_MPI
euler_zslab_cpu_mpi_SOURCES = euler_zslab_mpi_main.cpp 
euler_zslab_cpu_mpi_CXXFLAGS = $(MPI_CXXFLAGS) $(GM_CFLAGS) $(GM_FLAGS_SUPP) -I$(srcdir)/utils/mpiUtils  $(EXTRA_FLAGS)
euler_zslab_cpu_mpi_LDFLAGS = $(MPI_LDFLAGS) $(GM_LIBS)
euler_zslab_cpu_mpi_LDADD = hydro/libhydroCpu.la utils/config/libIniConfig.la utils/monitoring/libMonitoring.la utils/cnpy/libCNpy.la utils/mpiUtils/libMpiUtils.la
endif

### Double Precision - MPI ###
if USE_MPI_AND_DOUBLE_PRECISION
euler_zslab_cpu_mpi_double_SOURCES = euler_zslab_mpi_main.cpp 
euler_zslab_cpu_mpi_double_CXXFLAGS = -DUSE_DOUBLE $(MPI_CXXFLAGS) $(GM_CFLAGS) $(GM_FLAGS_SUPP) -I$(srcdir)/utils/mpiUtils  $(EXTRA_FLAGS)
euler_zslab_cpu_mpi_double_CPPFLAGS = -DUSE_DOUBLE $(AM_CPPFLAGS)
euler_zslab_cpu_mpi_double_LDFLAGS = $(MPI_LDFLAGS) $(GM_LIBS)
euler_zslab_cpu_mpi_double_LDADD = hydro/libhydroCpu_double.la utils/config/libIniConfig.la utils/monitoring/libMonitoring.la utils/cnpy/libCNpy.la utils/mpiUtils/libMpiUtils.la
endif

# =======================================
#
# Euler 3d on GPU with MPI+CUDA + Z-SLAB
#
# =======================================
if USE_CUDA
### SIMPLE PRECISION ###
if USE_MPI
BUILT_SOURCES += euler_zslab_mpi_main.cu
CLEANFILES    += euler_zslab_mpi_main.cu
nodist_euler_zslab_gpu_mpi_SOURCES = euler_zslab_mpi_main.cu 
# force CXX link
nodist_EXTRA_euler_zslab_gpu_mpi_SOURCES = dummy.cpp
euler_zslab_gpu_mpi_CPPFLAGS = $(CUDA_CFLAGS) $(MPI_CXXFLAGS) $(GM_CFLAGS) $(GM_FLAGS_SUPP) -I$(srcdir)/utils/mpiUtils $(EXTRA_FLAGS)
euler_zslab_gpu_mpi_CXXFLAGS = $(CUDA_CFLAGS) $(MPI_CXXFLAGS) $(GM_CFLAGS) $(GM_FLAGS_SUPP) -I$(srcdir)/utils/mpiUtils $(EXTRA_FLAGS)
euler_zslab_gpu_mpi_LDFLAGS  = $(CUDA_LIBS) $(GM_LIBS) hydro/libhydroGpu.la utils/config/libIniConfigGpu.la utils/monitoring/libMonitoringGpu.la utils/cnpy/libCNpy.la utils/mpiUtils/libMpiUtils.la $(MPI_LDFLAGS) 
endif

### DOUBLE PRECISION ###
if USE_MPI_AND_DOUBLE_PRECISION
BUILT_SOURCES += euler_zslab_mpi_main-double.cu
CLEANFILES    += euler_zslab_mpi_main-double.cu
nodist_euler_zslab_gpu_mpi_double_SOURCES = euler_zslab_mpi_main-double.cu 
# force CXX link
nodist_EXTRA_euler_zslab_gpu_mpi_double_SOURCES = dummy.cpp
euler_zslab_gpu_mpi_double_CXXFLAGS = -DUSE_DOUBLE $(CUDA_CFLAGS) $(MPI_CXXFLAGS) $(GM_CFLAGS) $(GM_FLAGS_SUPP) -I$(srcdir)/utils/mpiUtils $(EXTRA_FLAGS)
euler_zslab_gpu_mpi_double_CPPFLAGS = -DUSE_DOUBLE $(AM_CPPFLAGS)
euler_zslab_gpu_mpi_double_LDFLAGS  = $(CUDA_LIBS)  $(GM_LIBS) hydro/libhydroGpu_double.la utils/config/libIniConfigGpu.la utils/monitoring/libMonitoringGpu.la utils/cnpy/libCNpy.la utils/mpiUtils/libMpiUtils.la $(MPI_LDFLAGS)
endif

endif

###############################################
###############################################
## Some applications with a GLUT-based GUI
###############################################
###############################################
if HAVE_GLUT
bin_PROGRAMS += euler2d_cpu_glut

if USE_DOUBLE_PRECISION
bin_PROGRAMS += euler2d_cpu_glut_double
endif

# ==================================================================
#
# Euler 2d on CPU (Godunov-Riemann or Kurganov-Tadmor) with GLUT GUI
#
# ==================================================================
euler2d_cpu_glut_SOURCES = \
	glutGui/palettes.h \
	glutGui/GlutWindow.h \
	glutGui/GlutWindow.cpp \
	glutGui/GlutMaster.h \
	glutGui/GlutMaster.cpp \
	glutGui/HydroWindow.h \
	glutGui/HydroWindow.cpp \
	euler2d_main_glut.cpp
euler2d_cpu_glut_CXXFLAGS = -I$(srcdir)/glutGui $(GM_CFLAGS) $(GM_FLAGS_SUPP)
euler2d_cpu_glut_LDADD    = $(GM_LIBS) $(GLEW_LIBS) $(GLUT_LIBS) $(GL_LIBS) hydro/libhydroCpu.la utils/config/libIniConfig.la utils/monitoring/libMonitoring.la utils/cnpy/libCNpy.la $(PAPI_LDFLAGS)

if USE_DOUBLE_PRECISION
euler2d_cpu_glut_double_SOURCES = \
	glutGui/palettes.h \
	glutGui/GlutWindow.h \
	glutGui/GlutWindow.cpp \
	glutGui/GlutMaster.h \
	glutGui/GlutMaster.cpp \
	glutGui/HydroWindow.h \
	glutGui/HydroWindow.cpp \
	euler2d_main_glut.cpp
euler2d_cpu_glut_double_CXXFLAGS = -DUSE_DOUBLE -I$(srcdir)/glutGui $(GM_CFLAGS) $(GM_FLAGS_SUPP)
euler2d_cpu_glut_double_CPPFLAGS = -DUSE_DOUBLE -I$(srcdir)/glutGui $(AM_CPPFLAGS)
euler2d_cpu_glut_double_LDADD    = $(GM_LIBS) $(GLEW_LIBS) $(GLUT_LIBS)  $(GL_LIBS) hydro/libhydroCpu_double.la utils/config/libIniConfig.la utils/monitoring/libMonitoring.la utils/cnpy/libCNpy.la $(PAPI_LDFLAGS)
endif

# ==================================================================
#
# Euler 2d on GPU (Godunov-Riemann or Kurganov-Tadmor) with GLUT GUI
#
# ==================================================================
if USE_CUDA
BUILT_SOURCES += glutGui/GlutWindow.cu glutGui/GlutMaster.cu glutGui/HydroWindow.cu euler2d_main_glut.cu
CLEANFILES    += glutGui/GlutWindow.cu glutGui/GlutMaster.cu glutGui/HydroWindow.cu euler2d_main_glut.cu
bin_PROGRAMS  += euler2d_gpu_glut
euler2d_gpu_glut_SOURCES = \
	glutGui/minmax.cuh \
	glutGui/pbo.cuh \
	glutGui/GlutWindow.h \
	glutGui/GlutMaster.h \
	glutGui/HydroWindow.h

nodist_euler2d_gpu_glut_SOURCES = \
	glutGui/GlutWindow.cu \
	glutGui/GlutMaster.cu \
	glutGui/HydroWindow.cu \
	euler2d_main_glut.cu

nodist_EXTRA_euler2d_gpu_glut_SOURCES = dummy.cpp
euler2d_gpu_glut_CXXFLAGS = $(CUDA_CFLAGS) $(GM_CFLAGS) $(GM_FLAGS_SUPP)
euler2d_gpu_glut_LDFLAGS  = $(CUDA_LIBS) $(GLEW_LIBS) $(GLUT_LIBS) $(GL_LIBS) 
euler2d_gpu_glut_LDADD    = hydro/libhydroGpu.la utils/config/libIniConfigGpu.la utils/monitoring/libMonitoringGpu.la utils/cnpy/libCNpy.la $(GM_LIBS) 

endif

endif

###############################################
###############################################
## Some applications with a QT-based GUI
###############################################
###############################################
# if ENABLE_QTGUI
# bin_PROGRAMS += euler2d_cpu_qt

# # ===================================================
# # euler2d_cpu_qt
# # ===================================================
# euler2d_cpu_qt_SOURCES = \
# 	hydro/gl_util.h \
# 	qtGui/qtHydro2d/HydroWidget.h \
# 	qtGui/qtHydro2d/HydroWidget.cpp \
# 	qtGui/qtHydro2d/MainWindow.h \
# 	qtGui/qtHydro2d/MainWindow.cpp \
# 	qtGui/qtHydro2d/main.cpp 
# euler2d_cpu_qt_CXXFLAGS = $(GM_CFLAGS) $(GM_FLAGS_SUPP) -fPIC
# euler2d_cpu_qt_LDADD   = $(GM_LIBS) $(QT_LIBS) hydro/libhydroCpu.la utils/config/libIniConfig.la utils/monitoring/libMonitoring.la utils/cnpy/libCNpy.la $(PAPI_LDFLAGS)
# euler2d_cpu_qt_LDFLAGS = $(AM_LDFLAGS) $(GLEW_LIBS)

# moc_gen_sources = \
# 	qtGui/qtHydro2d/HydroWidget.moc.cpp \
# 	qtGui/qtHydro2d/MainWindow.moc.cpp

# gen_sources = $(moc_gen_sources) 

# nodist_euler2d_cpu_qt_SOURCES = $(gen_sources)

# BUILT_SOURCES += $(moc_gen_sources) 
# CLEANFILES += $(moc_gen_sources) 

# ===================================================
# euler2d_gpu_qt
# ===================================================
# if USE_CUDA
# BUILT_SOURCES += qtGui/qtHydro2d/main.cu
# CLEANFILES    += qtGui/qtHydro2d/main.cu
# bin_PROGRAMS += euler2d_gpu_qt
# euler2d_gpu_qt_SOURCES = \
# 	hydro/gl_util.h \
# 	qtGui/qtHydro2d/minmax.cuh \
# 	qtGui/qtHydro2d/pbo.cuh \
# 	qtGui/qtHydro2d/HydroWidgetGpu.h \
# 	qtGui/qtHydro2d/HydroWidgetGpu.cu \
# 	qtGui/qtHydro2d/HydroWidget.h \
# 	qtGui/qtHydro2d/HydroWidget.cpp \
# 	qtGui/qtHydro2d/MainWindow.h \
# 	qtGui/qtHydro2d/MainWindow.cpp
# nodist_euler2d_gpu_qt_SOURCES = qtGui/qtHydro2d/main.cu 
# nodist_EXTRA_euler2d_gpu_qt_SOURCES = dummy.cpp
# euler2d_gpu_qt_CXXFLAGS = $(CUDA_CFLAGS) $(GM_CFLAGS) $(GM_FLAGS_SUPP)
# euler2d_gpu_qt_LDFLAGS  = $(CUDA_LIBS) $(GLEW_LIBS) hydro/libhydroGpu.la utils/config/libIniConfigGpu.la utils/monitoring/libMonitoringGpu.la
# euler2d_gpu_qt_LDADD    = $(GM_LIBS) $(QT_LIBS) 
# gpu_moc_gen_sources = \
# 	qtGui/qtHydro2d/MainWindow.moc.cpp

# gpu_gen_sources = $(moc_gen_sources) 

# nodist_euler2d_gpu_qt_SOURCES += $(gpu_gen_sources)

# BUILT_SOURCES += $(moc_gen_sources) 
# endif

# endif

##########################################################
##########################################################
## some test programs
##########################################################
##########################################################

## test of the Riemann solver
bin_PROGRAMS += testRiemannHLLD_cpu testTrace_cpu
if USE_CUDA
bin_PROGRAMS += testRiemannHLLD_gpu
endif

# cpu sources for testRiemannHLLD_cpu
testRiemannHLLD_cpu_SOURCES = \
	testRiemannHLLD.cpp \
	hydro/gpu_macros.cpp \
	hydro/constants.cpp
testRiemannHLLD_cpu_CXXFLAGS = $(GM_CFLAGS) $(GM_FLAGS_SUPP) $(EXTRA_FLAGS)
testRiemannHLLD_cpu_LDADD = $(GM_LIBS) utils/config/libIniConfig.la utils/monitoring/libMonitoring.la

# cpu sources for testTrace_cpu
testTrace_cpu_SOURCES = \
	testTrace.cpp \
	hydro/gpu_macros.cpp \
	hydro/constants.cpp
testTrace_cpu_CXXFLAGS = $(GM_CFLAGS) $(GM_FLAGS_SUPP) $(EXTRA_FLAGS)
testTrace_cpu_LDADD = $(GM_LIBS) utils/config/libIniConfig.la utils/monitoring/libMonitoring.la


if USE_DOUBLE_PRECISION
bin_PROGRAMS += testRiemannHLLD_cpu_double testTrace_cpu_double
testRiemannHLLD_cpu_double_SOURCES = \
	testRiemannHLLD.cpp \
	hydro/gpu_macros.cpp \
	hydro/constants.cpp
testRiemannHLLD_cpu_double_CXXFLAGS = -DUSE_DOUBLE $(GM_CFLAGS) $(GM_FLAGS_SUPP) $(EXTRA_FLAGS)
testRiemannHLLD_cpu_double_LDADD = $(GM_LIBS) utils/config/libIniConfig.la utils/monitoring/libMonitoring.la

testTrace_cpu_double_SOURCES = \
	testTrace.cpp \
	hydro/gpu_macros.cpp \
	hydro/constants.cpp
testTrace_cpu_double_CXXFLAGS = -DUSE_DOUBLE $(GM_CFLAGS) $(GM_FLAGS_SUPP) $(EXTRA_FLAGS)
testTrace_cpu_double_LDADD = $(GM_LIBS) utils/config/libIniConfig.la utils/monitoring/libMonitoring.la
endif

# gpu sources for testRiemannHLLD_gpu
if USE_CUDA
BUILT_SOURCES += testRiemannHLLD.cu
CLEANFILES    += testRiemannHLLD.cu
nodist_testRiemannHLLD_gpu_SOURCES = \
	testRiemannHLLD.cu \
	hydro/gpu_macros.cu \
	hydro/constants.cu
nodist_EXTRA_testRiemannHLLD_gpu_SOURCES = dummy.cpp
testRiemannHLLD_gpu_CPPFLAGS = $(CUDA_CFLAGS) $(EXTRA_FLAGS)
testRiemannHLLD_gpu_LDADD = $(CUDA_LIBS) $(GM_LIBS) utils/config/libIniConfig.la utils/monitoring/libMonitoring.la

if USE_DOUBLE_PRECISION
bin_PROGRAMS += testRiemannHLLD_gpu_double
BUILT_SOURCES += testRiemannHLLD-double.cu
CLEANFILES    += testRiemannHLLD-double.cu
nodist_testRiemannHLLD_gpu_double_SOURCES = \
	testRiemannHLLD-double.cu \
	hydro/gpu_macros.cu \
	hydro/constants.cu
nodist_EXTRA_testRiemannHLLD_gpu_double_SOURCES = dummy.cpp
testRiemannHLLD_gpu_double_CPPFLAGS = -DUSE_DOUBLE $(CUDA_CFLAGS) $(EXTRA_FLAGS)
testRiemannHLLD_gpu_double_LDADD = $(CUDA_LIBS) $(GM_LIBS) utils/config/libIniConfig.la utils/monitoring/libMonitoring.la
endif

endif
