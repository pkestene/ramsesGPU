/**
 * \file HydroWindow.h
 * \brief
 * Top level class for displaying a Hydro simulation.
 * Inspired by DemoWindow.h
 *
 * \author P. Kestener
 * \date 23/02/2010
 *
 * $Id: HydroWindow.h 1784 2012-02-21 10:34:58Z pkestene $
 */


#ifndef HYDRO_WINDOW_H_
#define HYDRO_WINDOW_H_

#include "GlutMaster.h"

// application header
#include <ConfigMap.h>
#include <HydroRunBase.h>
#include <HydroRunGodunov.h>
#include <HydroRunKT.h>

using hydroSimu::HydroRunBase;
using hydroSimu::HostArray;
#ifdef __CUDACC__
using hydroSimu::DeviceArray;
#endif // __CUDACC__

// CUDA / OpenGl interoperability
#ifdef __CUDACC__
#include "cutil_inline.h"
#include <cuda_gl_interop.h>
#include "pbo.cuh"
#include "minmax.cuh"
#endif // __CUDACC__

// for HDF5 file format output
#ifdef USE_HDF5
#include <hdf5.h>
#endif // USE_HDF5


#define I2D(nx,i,j) (((nx)*(j)) + i)
//#define NBCOLORS (236)
#define NBCOLORS (256)

/** use the grapgics API available from CUDA >= 3.0 */
//#define USE_CUDA3

/**
 * \class HydroWindow HydroWindow.h
 * \brief a specialized version of a GlutWindow dedicated to display
 * and interaction with a running hydrodynamics simulation (display
 * update as results are computed).
 */
class HydroWindow : public GlutWindow {
public:
  
  int          nx,ny;
  int          ghostWidth;
  int          nxg, nyg;
  int          initPositionX, initPositionY;
  bool         ghostIncluded;
  bool         unsplitEnabled;

  // OpenGL pixel buffer object and texture
  GLuint gl_PBO, gl_Tex;
#ifdef __CUDACC__
#  ifdef USE_CUDA3
  struct cudaGraphicsResource* cuda_PBO;
#  endif // USE_CUDA3
#endif // __CUDACC__

  // simulation data
  ConfigMap param;
  HydroRunBase * hydroRun;
  real_t t;
  real_t dt;
  int nStep;
  int nbStepToDo;
  bool animate;                 //!< if true perform continuous computation
  int useColor;                 //!< enable color (default is 0 : gray scale)
  int manualContrast;           //!< enable manual contrast
			        //! (minvar and maxvar are not
			        //! computed at each time step
			        //! but controlled through
			        //! keyboard shortcut 'a' and 'b'
  int currentRiemannConfigNb;

  // colormap
  float *cmap;                  //!< pointer to colormap array
			        //!  defined in palettes.h
  float *cmap_der;              //!< array needed for color interpolation
  void initColormap();               
  unsigned int *plot_rgba;       //!< array of pixel color
  real_t minvar;                //!< minimum value used to rescale data
  real_t maxvar;                //!< maximum value used to rescale data
  int displayVar;               //!< integer identifying component to display
  uint minmaxBlockCount;

#ifdef __CUDACC__
  float *d_cmap;               //!< indexed colormap data on device
  float *d_cmap_der;           //!< indexed colormap derivative (interpolation)
  unsigned int *plot_rgba_pbo; //!< data to plot (physicaly in the Pixel buffer Object)
#endif // __CUDACC__

protected:
  void createPBO();
  void deletePBO();
  void createTexture();
  void deleteTexture();
  //! bufferChoice allows to choose between h_U and h_U2 (CPU) or
  //! between d_U and d_U2 (GPU)
  void render(int bufferChoice);

private:
  // Data Arrays
#ifdef __CUDACC__
  HostArray<real2_t>   h_minmax;
  DeviceArray<real2_t> d_minmax;
#endif // __CUDACC__

  
public:
  HydroWindow(HydroRunBase* _hydroRun,
	      GlutMaster * glutMaster,
	      int setInitPositionX, 
	      int setInitPositionY,
	      const char* title,
	      ConfigMap & _param);
  ~HydroWindow();
  
  void CallBackDisplayFunc(void);
  void CallBackReshapeFunc(int w, int h);   
  void CallBackIdleFunc(void);
  void CallBackKeyboardFunc(unsigned char key, int x, int y);

  void startSimulation(GlutMaster * glutMaster);
  void computeOneStepAndDisplay();

#ifdef __CUDACC__
  void convertDataForPlotting(int _useColor, int bufferChoice);
#endif // __CUDACC__

  //! compute minvar and maxvar (simple reduction).
  void computeMinMax(HostArray<real_t> &U, int size, int iVar);
#ifdef __CUDACC__
  //! this is the GPU version which wraps the CUDA kernel call.
  void computeMinMax(DeviceArray<real_t> &U, int size, int iVar);
#endif // __CUDACC__

};

#endif // HYDRO_WINDOW_H_
