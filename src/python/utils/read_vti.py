'''
A simple example showing how to read a .vti file (vtkImageData)
a convert it to a numpy array.
'''
import numpy as np
import matplotlib.pyplot as plt

import vtk

from vtk.util import numpy_support


#
# filename is a VTI file
#
def vti_to_numpy(filename, fieldName='density'):
    
    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(filename)
    reader.Update()

    # get vtkImageData
    imageData = reader.GetOutput()

    sr = imageData.GetScalarRange()
    print("scalar range {}".format(sr))

    # get dimensions tuple
    dims = imageData.GetDimensions()
    print("dims {}".format(dims))
    
    # get vtk data
    vtk_data = imageData.GetPointData().GetArray(fieldName)

    # convert to numpy array
    numpy_data = numpy_support.vtk_to_numpy(vtk_data)

    #numpy_data = numpy_data.reshape(dims[0], dims[1], dims[2])
    numpy_data = numpy_data.reshape(dims[1], dims[0])
    print("shape in reader {}".format(numpy_data.shape))

    #numpy_data = numpy_data.transpose(2,1,0)
    
    return numpy_data

if __name__ == '__main__':

    file1 = "jet2d_cpu_0000090.vti"
    file2 = "jet2d_cpu_v1_0000090.vti"
    

    d1 = vti_to_numpy(file1)
    d2 = vti_to_numpy(file2)

    diff=d2-d1
    print("L2 diff {}".format(np.linalg.norm(diff,ord='fro')))
    
    #shape = d1.shape
    #print("shape {}".format(shape))
    
    # get 2D array
    #d1 = d1.reshape(shape[0], shape[1])
    #d2 = d2.reshape(shape[0], shape[1])
    
    plt.subplot(131)
    plt.imshow(d1)
    plt.subplot(132)
    plt.imshow(d2)
    plt.subplot(133)
    plt.imshow(d2-d1)

    plt.show()
