/**
 * \file HydroWidgetGpu.h
 * \brief
 * OpenGL Widget to display HydroRun simulation (CUDA version).
 *
 * \author Pierre Kestener
 * \date 8 Oct 2010
 */
#ifndef HYDRO_WIDGET_GPU_H_
#define HYDRO_WIDGET_GPU_H_

#include "HydroWidget.h"

using hydroSimu::DeviceArray;

/** use the graphics OpenGL/CUDA interoperability API available from CUDA >= 3.0 */
//#define USE_CUDA3

/**
 * \class HydroWidgetGpu HydroWidgetGpu.h
 * \brief This is a specialization of HydroWidget to handle display of
 * GPU computation results.
 */
class HydroWidgetGpu : public HydroWidget
{

public:
  HydroWidgetGpu(ConfigMap& _param, HydroRunBase* _hydroRun, QWidget *parent = 0);
  virtual ~HydroWidgetGpu();

  /** indexed colormap data on device*/
  unsigned int *cmap_rgba_device;
  /** data to plot (physicaly in the Pixel buffer Object) */
  unsigned int *plot_rgba_pbo;
  uint minmaxBlockCount;

public:
  virtual void initColormap();

protected:
  // Data Arrays
  HostArray<real2_t>   h_minmax;
  DeviceArray<real2_t> d_minmax;
  void convertDataForPlotting(int _useColor);
  virtual void computeMinMax(real_t *U, int size, int iVar);

  virtual void createPBO();
  virtual void deletePBO();
  virtual void render();
  virtual void initializeGL();

private:
#  ifdef USE_CUDA3
  struct cudaGraphicsResource* cuda_PBO;
#  endif // USE_CUDA3

}; // class HydroWidgetGpu

#endif // HYDRO_WIDGET_GPU_H_
