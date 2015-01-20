/**
 * \file vtk_inc.h
 * \brief Gather all vtk related headers here.
 *
 * \author Pierre Kestener.
 * \date 8-July-2010
 *
 * $Id: vtk_inc.h 1784 2012-02-21 10:34:58Z pkestene $
 */
#ifndef VTK_INC_H_
#define VTK_INC_H_

// the following symbol prevents some nasting warning about deprecated
// strstream header
#define VTK_EXCLUDE_STRSTREAM_HEADERS

#include <vtkSmartPointer.h>
#include <vtkImageData.h>
#include <vtkFloatArray.h>
#include <vtkDoubleArray.h>
#include <vtkPointData.h>
#include <vtkXMLImageDataWriter.h>
#include <vtkImageWriter.h>
#include <vtkImageReader.h>
#include <vtkDataArray.h>

#endif // VTK_INC_H_
