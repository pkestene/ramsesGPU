ACLOCAL_AMFLAGS = -I m4

SUBDIRS = src test

EXTRA_DIST = autogen.sh am/cudalt.py

dist_data_DATA = \
	data/jet2d_cpu.ini \
	data/jet2d_gpu.ini \
	data/riemann2d.ini \
	data/implode2d.ini \
	data/blast2d.ini

# Doxygen support
include $(top_srcdir)/am/am_include_doxygen.am

# clean generated files
MAINTAINERCLEANFILES = $(srcdir)/Makefile.in 

