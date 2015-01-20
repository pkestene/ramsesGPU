#!/usr/bin/env python

"""Visualize hdf5 data from a RAMSES-GPU run.

"""

# Standard library imports
import numpy as np

import h5py

import sys

# Enthought library imports
from mayavi.scripts import mayavi2
from mayavi.sources.array_source import ArraySource
from mayavi.modules.outline import Outline
from mayavi.modules.image_plane_widget import ImagePlaneWidget


def read_data(filename,fieldName='x-velocity'):
    """Read hdf5 data from RAMSES-GPU simulation run"""

    # open file
    f=h5py.File(filename,'r')

    # get numpy array
    dataValue=f[fieldName].value

    # close file
    f.close()

    # return array
    return dataValue


@mayavi2.standalone
def view_data(filename,fieldName='x-velocity'):
    """Visualize a 3D numpy array in mayavi2.
    """
    # 'mayavi' is always defined on the interpreter.
    mayavi.new_scene()
    # Make the data and add it to the pipeline.
    data = read_data(filename,fieldName)
    src = ArraySource(transpose_input_array=False)
    src.scalar_data = data
    mayavi.add_source(src)
    # Visualize the data.
    o = Outline()
    mayavi.add_module(o)
    ipw = ImagePlaneWidget()
    mayavi.add_module(ipw)
    ipw.module_manager.scalar_lut_manager.show_scalar_bar = True
    ipw.module_manager.scalar_lut_manager.data_name = fieldName

    ipw_y = ImagePlaneWidget()
    mayavi.add_module(ipw_y)
    ipw_y.ipw.plane_orientation = 'y_axes'

    ipw_z = ImagePlaneWidget()
    mayavi.add_module(ipw_z)
    ipw_z.ipw.plane_orientation = 'z_axes'


if __name__ == '__main__':

    # get filename
    if (len(sys.argv) > 1):
        filename=sys.argv[1]

    # get array data name
    dataname=""
    if (len(sys.argv) > 2):
        dataname=sys.argv[2]

    # visualize it !
    if dataname != "" :
        view_data(filename,dataname)
    else:
        view_data(filename)

