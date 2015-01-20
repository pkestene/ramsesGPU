# clean generated files
MAINTAINERCLEANFILES = $(srcdir)/Makefile.in

AUTOMAKE_OPTIONS = subdir-objects

SUBDIRS = config monitoring
if USE_MPI
SUBDIRS += mpiUtils
endif
SUBDIRS += cnpy IO
