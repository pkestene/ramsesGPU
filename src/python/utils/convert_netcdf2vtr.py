#!/usr/bin/env python

"""
Convert Netcdf file to python.

"""

# Standard library imports
from __future__ import division
import sys

# numerical packages
import numpy as np

# HDF5 package
import h5py as h5

# NetCDF package
import Scientific.IO.NetCDF as nc

# write numpy array into VTK file
from evtk.hl import gridToVTK


#
# ###################################################
#
def read_data(filename, fieldName):

    if filename.endswith('.h5'):
        return read_h5(filename, fieldName)
    elif filename.endswith('.nc'):
        return read_nc(filename, fieldName)
    else:
        print 'Wrong filename (only hdf5 and netcdf supported)'

#
# ###################################################
#
def read_h5(filename,fieldName='x-velocity'):
    """Read hdf5 data from RAMSES-GPU simulation run using h5py module.

    Possible value for fieldName : density, energy,
    magnetic_field_x, magnetic_field_y, magnetic_field_z,
    momentum_x, momentum_y, momentum_z

    return numpy array of corresponding field.
    """

    # open file
    f=h5.File(filename,'r')

    # TODO test if fieldName is in f.keys()

    # get numpy array
    data=f[fieldName].value

    # get total time
    totalTime = f.attrs['total time']

    # close file
    f.close()

    # print '%s read; returning field %s (%d,%d,%d)' % (fileName, fieldName,
    #                                                   data.shape[0],
    #                                                   data.shape[1],
    #                                                   data.shape[2])

    # return array
    #return data, totalTime
    return data

#
# ###################################################
#  
def read_nc(fileName,fieldName='rho'):
    """ReadNetCDF data from RAMSES-GPU simulation run using Scientific.IO.NetCDF module.

    Possible value for fieldName : rho, E, rho_vx, rho_vy, rho_vz,
    Bx, By, Bz

    return numpy array of corresponding field.
    """

    f=nc.NetCDFFile(fileName,'r')

    # TODO test if fieldName is in f.keys()
    data = f.variables[fieldName].getValue()

    # get total time
    totalTime = getattr(f, 'total time')

    f.close()

    # print '%s read; returning field %s (%d,%d,%d)' % (fileName, fieldName,
    #                                                   data.shape[0],
    #                                                   data.shape[1],
    #                                                   data.shape[2])

    #return data, totalTime
    return data



#
# ######################################
#
def toVtr(filePrefix):

    iFile=filePrefix+'.nc'

    # geometry
    rho        = read_data(iFile, 'rho')
    nx,ny,nz   = rho.shape
    lx, ly, lz = 1.0*nx, 1.0*ny, 1.0*nz
    dx, dy, dz = lx/nx, ly/ny, lz/nz
    ncells     = nx * ny * nz

    # Coordinates
    x = np.arange(0, lx + 0.1*dx, dx, dtype='float64')
    y = np.arange(0, ly + 0.1*dy, dy, dtype='float64')
    z = np.arange(0, lz + 0.1*dz, dz, dtype='float64')

    # read input data
    rho    = read_data(iFile, 'rho')
    rho_vx = read_data(iFile, 'rho_vx')
    rho_vy = read_data(iFile, 'rho_vy')
    rho_vz = read_data(iFile, 'rho_vz')
    Bx     = read_data(iFile, 'Bx')
    By     = read_data(iFile, 'By')
    Bz     = read_data(iFile, 'Bz')
    E      = read_data(iFile, 'E')

    #
    # write output file
    #
    
    # write file
    gridToVTK(filePrefix, x, y, z, cellData = {"rho" : rho,
                                               "E" : E,
                                               "rho_vx" : rho_vx,
                                               "rho_vy" : rho_vy,
                                               "rho_vz" : rho_vz,
                                               "Bx" : Bx,
                                               "By" : By,
                                               "Bz" : Bz})
    
    
#
# ######################################
#
if __name__ == '__main__':

    # get prefix
    if len(sys.argv)>1:
        prefix=sys.argv[1]

    # get start
    if len(sys.argv)>2:
        iStart=int(sys.argv[2])

    # get stop
    if len(sys.argv)>3:
        iStop=int(sys.argv[3])

    # get delta index
    if len(sys.argv)>4:
        deltaI=int(sys.argv[4])

    i=iStart
    while i<=iStop:

        iStr='%07d' % i
        
        filePrefix=prefix+iStr+'_IX'
        print 'Converting '+filePrefix+'.nc'+' into VTR file format ...'
        toVtr(filePrefix)

        filePrefix=prefix+iStr+'_IY'
        print 'Converting '+filePrefix+'.nc'+' into VTR file format ...'
        toVtr(filePrefix)

        filePrefix=prefix+iStr+'_IZ'
        print 'Converting '+filePrefix+'.nc'+' into VTR file format ...'
        toVtr(filePrefix)

        i += deltaI



