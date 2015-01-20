include $(top_srcdir)/am/autotroll.am

AUTOMAKE_OPTIONS = subdir-objects

AM_CPPFLAGS = $(QT_CPPFLAGS)
AM_CXXFLAGS = $(QT_CXXFLAGS)

# ===================================================
bin_PROGRAMS = qtHelloWorld

# ===================================================
qtHelloWorld_SOURCES = \
	glwidget.cpp \
	main.cpp \
	window.cpp
qtHelloWorld_LDADD = $(QT_LIBS)

moc_gen_sources = \
	glwidget.moc.cpp \
	window.moc.cpp

gen_sources = $(moc_gen_sources) 

nodist_qtHelloWorld_SOURCES = $(gen_sources)

BUILT_SOURCES = $(moc_gen_sources) 

# ===================================================

# clean generated files
MAINTAINERCLEANFILES    = $(srcdir)/Makefile.in

clean-local:
	rm -f core.* *~
