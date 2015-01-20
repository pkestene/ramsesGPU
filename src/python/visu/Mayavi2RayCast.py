#!/usr/bin/env python
"""This is a simple Mayavi2 script to do volume rendering using Mayavi2.
"""

##
# \file Mayavi2RayCast.py
# \brief This is a simple Mayavi2 script to do volume rendering using Mayavi2.
# \author P. Kestener

# Standard library imports
from os.path import join, abspath, dirname
import sys

# Mayavi imports.
from enthought.mayavi.scripts import mayavi2
from enthought.mayavi.sources.api import VTKXMLFileReader
from enthought.mayavi.modules.api import Surface, Outline, Volume

@mayavi2.standalone
def main():
    mayavi.new_scene()

    # Read the data:
    r = VTKXMLFileReader()
    filename = sys.argv[1]
    #'implode3d_gpu_d_0000380.vti'
    r.initialize(filename)
    mayavi.add_source(r)

    # Simple outline for the data.
    o = Outline()
    mayavi.add_module(o)

    # load the Volume rendering module
    v = Volume()
    mayavi.add_module(v)

if __name__ == "__main__":
    if (len(sys.argv) > 1):
        main()
    else:
        print "you must provide input filename (vti extension)."
        sys.exit("Execution failed.")
