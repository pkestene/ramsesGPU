# clean generated files
MAINTAINERCLEANFILES = $(srcdir)/Makefile.in

AUTOMAKE_OPTIONS = subdir-objects

# ===================================================
# Some convenience library
# ===================================================

CXX = $(MPICXX)

noinst_LTLIBRARIES = libMpiUtils.la

libMpiUtils_la_SOURCES = \
	GlobalMpiSession.h \
	GlobalMpiSession.cpp \
	TestForException.h \
	ErrorPolling.h \
	ErrorPolling.cpp \
	common_config.h \
	MpiComm.h \
	MpiComm.cpp \
	MpiCommCart.h \
	MpiCommCart.cpp
libMpiUtils_la_CPPFLAGS = $(AM_CPPFLAGS) 
# -Wno-unused-function -Wno-unused-parameter

