#!/usr/bin/env python
"""A simple to do offscreen rendering
or the Mayavi Envisage application and do off screen rendering. 

It can be run as::

    $ python offscreen_render.py
"""

# Author: Prabhu Ramachandran <prabhu@aero.iitb.ac.in>
# Copyright (c) 2007, Enthought, Inc.
# License: BSD Style.

##
# \file offscreen_render.py
# Mayavi2-based offscreen rendering script adapted to hydro simulation 
# by P. Kestener (Sept. 2010) 

from os.path import join, abspath, dirname
import sys

# The offscreen Engine.
from enthought.mayavi.api import OffScreenEngine

# Usual MayaVi imports
from enthought.mayavi.sources.api import VTKXMLFileReader
from enthought.mayavi.modules.api import Outline, ScalarCutPlane, ImagePlaneWidget, Volume

def offscreen_render(filename, do_volume_rendering):
    # Create the MayaVi offscreen engine and start it.
    e = OffScreenEngine()
    # Starting the engine registers the engine with the registry and
    # notifies others that the engine is ready.
    e.start()

    # Create a new scene.
    win = e.new_scene()

    # Now setup a normal MayaVi pipeline.
    src = VTKXMLFileReader()
    src.initialize(filename)

    e.add_source(src)

    # outline (make it black, enlarge width)
    o = Outline()
    e.add_module(o)
    o.actor.property.color = (0,0,0)
    o.actor.property.line_width = 1.5

    if (do_volume_rendering == 0):
        # image plane widgets
        ipwx = ImagePlaneWidget()
        e.add_module(ipwx)

        ipwy = ImagePlaneWidget()
        e.add_module(ipwy)
        ipwy.ipw.plane_orientation = 'y_axes'
        
        ipwz = ImagePlaneWidget()
        e.add_module(ipwz)
        ipwz.ipw.plane_orientation = 'z_axes'
    else:
        # volume rendering
        volren = Volume()
        e.add_module(volren)

    win.scene.isometric_view()

    # Set the view.
    s = e.current_scene
    cam = s.scene.camera
    #cam.azimuth(45)
    #cam.elevation(15)
    cam.zoom(1.0)
    cam.view_up = [-0.1, 0.89, -0.546]
    
    # Change the size argument to anything you want.
    win.scene.save(filename+'.jpg', size=(600, 600))
    
if __name__ == '__main__':

    for i in range(100, 1500, 20):
        filename = 'jet3d_gpu_d_'+'%07d.vti' % i
        print 'processing file '+filename
        offscreen_render(filename,1)
    
    # to convert to a mpg animation:
    # convert -delay 50 jet3d_gpu_d_*.jpg jet3d_gpu_d.mpg
