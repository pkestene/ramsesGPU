AUTOMAKE_OPTIONS = subdir-objects

# clean generated files
MAINTAINERCLEANFILES = $(srcdir)/Makefile.in

CXX=@MPICXX@
AM_CPPFLAGS = -Wno-unused-parameter -Wno-unused-function

bin_PROGRAMS =

# Some MPI test programs
if USE_MPI
bin_PROGRAMS += \
	testMpiHello \
	testMpiTopology_2D_C_API \
	testMpiTopology_3D_C_API \
	testMpiTopology_2D_Cpp_API \
	testMpiTopology_3D_Cpp_API \
	testMpiHelloGlobalSession \
	testMpiTopology_2D

testMpiHello_SOURCES = testMpiHello.cpp
testMpiTopology_2D_C_API_SOURCES = testMpiTopology_2D_C_API.cpp
testMpiTopology_3D_C_API_SOURCES = testMpiTopology_3D_C_API.cpp
testMpiTopology_2D_Cpp_API_SOURCES = testMpiTopology_2D_Cpp_API.cpp
testMpiTopology_3D_Cpp_API_SOURCES = testMpiTopology_3D_Cpp_API.cpp

testMpiHelloGlobalSession_SOURCES  = testMpiHelloGlobalSession.cpp
testMpiHelloGlobalSession_CPPFLAGS = $(AM_CPPFLAGS) -I$(top_srcdir)/src/utils/mpiUtils
testMpiHelloGlobalSession_LDFLAGS  = $(AM_LDFLAGS) ../../src/utils/mpiUtils/libMpiUtils.la

testMpiTopology_2D_SOURCES  = testMpiTopology_2D.cpp
testMpiTopology_2D_CPPFLAGS = $(AM_CPPFLAGS) -I$(top_srcdir)/src/utils/mpiUtils
testMpiTopology_2D_LDFLAGS  = $(AM_LDFLAGS) ../../src/utils/mpiUtils/libMpiUtils.la

if HAVE_VTK
bin_PROGRAMS += testVtkXMLPImageDataWriter
testVtkXMLPImageDataWriter_SOURCES = testVtkXMLPImageDataWriter.cpp
testVtkXMLPImageDataWriter_CPPFLAGS = $(AM_CPPFLAGS) $(VTK_CPPFLAGS)
testVtkXMLPImageDataWriter_LDFLAGS = $(AM_LDFLAGS) $(VTK_LDFLAGS) $(VTK_LIBS)
endif

# Some MPI test programs using Boost C++ API
# if USE_MPI_BOOST
# bin_PROGRAMS += testMpiHelloBoost

# testMpiHelloBoost_SOURCES  = testMpiHelloBoost.cpp
# testMpiHelloBoost_CPPFLAGS = $(AM_CPPFLAGS) $(BOOST_CPPFLAGS) -Wno-ignored-qualifiers
# testMpiHelloBoost_LDFLAGS  = $(AM_LDFLAGS)  $(BOOST_LDFLAGS) $(BOOST_MPI_LIB)

# endif

endif

