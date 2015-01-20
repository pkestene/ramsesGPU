#!/usr/bin/env python

##
# \file VtkRayCast.py
# \brief This is a simple volume rendering example that uses a
# vtkVolumeRayCast mapper

#
# Modified by P. Kestener :
# - handle vtkImageData type
# - custom colorTransfertFunction
#


import sys

import vtk
from vtk.util.misc import vtkGetDataRoot
VTK_DATA_ROOT = vtkGetDataRoot()


def CheckAbort(obj, event):
    if obj.GetEventPending() != 0:
        obj.SetAbortRender(1)

def doRayCast(filename):
    
    # Create the standard renderer, render window and interactor
    ren = vtk.vtkRenderer()
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)

    # Create the reader for the data
    #reader = vtk.vtkStructuredPointsReader()
    #reader.SetFileName(VTK_DATA_ROOT + "/Data/ironProt.vtk")
    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(filename)

    # shift and scale
    scaler = vtk.vtkImageShiftScale()
    scaler.SetScale(255)
    scaler.SetInputConnection(reader.GetOutputPort())
    scaler.SetOutputScalarTypeToUnsignedChar()

    # Create transfer mapping scalar value to opacity
    opacityTransferFunction = vtk.vtkPiecewiseFunction()
    opacityTransferFunction.AddPoint(20, 0.0)
    opacityTransferFunction.AddPoint(255, 0.2)

    # Create transfer mapping scalar value to color
    colorTransferFunction = vtk.vtkColorTransferFunction()
    # colorTransferFunction.AddRGBPoint(0.0, 0.0, 0.0, 0.0)
    # colorTransferFunction.AddRGBPoint(64.0, 1.0, 0.0, 0.0)
    # colorTransferFunction.AddRGBPoint(128.0, 0.0, 0.0, 1.0)
    # colorTransferFunction.AddRGBPoint(192.0, 0.0, 1.0, 0.0)
    # colorTransferFunction.AddRGBPoint(255.0, 0.0, 0.2, 0.0)
    # colorTransferFunction.AddRGBSegment(0.0, 1.0, 1.0, 1.0, 255.0, 1.0, 1.0, 1.0)
    colorTransferFunction.AddRGBPoint(80.0 , 0.0, 0.0, 0.0 )
    colorTransferFunction.AddRGBPoint(120.0, 0.0, 0.0, 1.0)
    colorTransferFunction.AddRGBPoint(160.0, 1.0, 0.0, 0.0)
    colorTransferFunction.AddRGBPoint(200.0, 0.0, 1.0, 0.0)
    colorTransferFunction.AddRGBPoint(255.0, 0.0, 1.0, 1.0)

    # The property describes how the data will look
    volumeProperty = vtk.vtkVolumeProperty()
    volumeProperty.SetColor(colorTransferFunction)
    volumeProperty.SetScalarOpacity(opacityTransferFunction)
    volumeProperty.ShadeOn()
    volumeProperty.SetInterpolationTypeToLinear()
    
    # The mapper / ray cast function know how to render the data
    compositeFunction = vtk.vtkVolumeRayCastCompositeFunction()
    volumeMapper = vtk.vtkVolumeRayCastMapper()
    volumeMapper.SetVolumeRayCastFunction(compositeFunction)
    # volumeMapper.SetInputConnection(reader.GetOutputPort())
    volumeMapper.SetInputConnection(scaler.GetOutputPort())

    # The volume holds the mapper and the property and
    # can be used to position/orient the volume
    volume = vtk.vtkVolume()
    volume.SetMapper(volumeMapper)
    volume.SetProperty(volumeProperty)
    
    ren.AddVolume(volume)
    ren.SetBackground(1, 1, 1)
    renWin.SetSize(600, 600)
    renWin.Render()

    # set camera properties
    ren.GetActiveCamera().SetFocalPoint(1, 0, 0)
    ren.GetActiveCamera().SetPosition(20, 0, 0)
    ren.GetActiveCamera().SetViewUp(0, 0, 10)
    ren.ResetCamera()
    ren.GetActiveCamera().Azimuth(180)
    ren.GetActiveCamera().Elevation(30)
    ren.GetActiveCamera().Dolly(1.2)
    ren.ResetCameraClippingRange()

    renWin.AddObserver("AbortCheckEvent", CheckAbort)

    # screenshot
    #w2if = vtk.vtkWindowToImageFilter()
    #w2if.SetInput(renWin)
    #w2if.Update()

    #writer = vtk.vtkPNGWriter()
    #writer.SetInput(w2if.GetOutput())
    #writer.SetFileName(filename+'.png')
    #writer.Write()


    iren.Initialize()
    renWin.Render()
    iren.Start()

if __name__ == "__main__":

    # get input vtkImageData (vti extension) filename from argv[1]
    if (len(sys.argv) > 1):
        vtiFilename = sys.argv[1]
    else:
        print "you must provide input filename (vti extension)."
        sys.exit("Execution failed.")

    doRayCast(vtiFilename)

    # if you want to do multiple rendering on a data time series
    #for i in range(0, 400, 20):
    #    vtiFilename = vtiFilenamePrefix + '%03d.vti' % i
    #    doRayCast(vtiFilename)
