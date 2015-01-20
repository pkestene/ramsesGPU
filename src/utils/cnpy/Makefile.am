AUTOMAKE_OPTIONS = subdir-objects

# clean generated files
MAINTAINERCLEANFILES = $(srcdir)/Makefile.in

# ===================================================
# Some convenience library
# ===================================================

noinst_LTLIBRARIES = libCNpy.la

libCNpy_la_SOURCES = cnpy.cpp cnpy.h zlib/crc32.c
libCNpy_la_CPPFLAGS = $(AM_CPPFLAGS) -I$(srcdir)

# some test program
bin_PROGRAMS = cnpy_example

# please note that
# zlib support (crc32 routine) is taken from official zlib implementation
# to avoid external dependency.

cnpy_example_SOURCES = cnpy_example.cpp
cnpy_example_CPPFLAGS =
cnpy_example_LDFLAGS = libCNpy.la

