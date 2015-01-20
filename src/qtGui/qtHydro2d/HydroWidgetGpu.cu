/**
 * \file HydroWidgetGpu.cu
 * \brief Implements class HydroWidgetGpu.
 *
 * \author Pierre Kestener
 * \date 8 Oct 2010
 */
#include <GL/glew.h>

// application header
#include "HydroWidgetGpu.h"
#include <algorithm>

// CUDA / OpenGl interoperability
#include "cutil_inline.h"
#include <cuda_gl_interop.h>
#include "pbo.cuh"
#include "minmax.cuh"

#include "gl_util.h"

// =======================================================
// =======================================================
HydroWidgetGpu::HydroWidgetGpu(ConfigMap& _param, HydroRunBase* _hydroRun,  QWidget *parent)
  : minmaxBlockCount(192),
    h_minmax(),
    d_minmax(),
    HydroWidget(_param, _hydroRun, parent)
{

# ifdef USE_CUDA3
  cuda_PBO = NULL;
# endif // USE_CUDA3

  minmaxBlockCount = std::min(minmaxBlockCount, blocksFor(hydroRun->getDataHost().section(), MINMAX_BLOCK_SIZE * 2));

  h_minmax.allocate(make_uint3(minmaxBlockCount, 1, 1));
  d_minmax.allocate(make_uint3(minmaxBlockCount, 1, 1));


} // HydroWidgetGpu::HydroWidgetGpu

// =======================================================
// =======================================================
HydroWidgetGpu::~HydroWidgetGpu()
{

  CUDA_SAFE_CALL( cudaFree(cmap_rgba_device) );

} // HydroWidgetGpu::~HydroWidgetGpu

// =======================================================
// =======================================================
void HydroWidgetGpu::computeMinMax(real_t *U, int size, int iVar)
{
  minmax_kernel<MINMAX_BLOCK_SIZE><<<
    minmaxBlockCount, 
    MINMAX_BLOCK_SIZE, 
    MINMAX_BLOCK_SIZE*sizeof(real2_t)>>>(hydroRun->getData().data(), 
					 d_minmax.data(), 
					 hydroRun->getData().section(),
					 hydroRun->getData().pitch(),
					 hydroRun->getData().dimx(),
					 iVar);
  d_minmax.copyToHost(h_minmax);
  real2_t* minmax = h_minmax.data();
  
  maxvar = -3.40282347e+38f;
  minvar = 3.40282347e+38f;

  for(uint i = 0; i < minmaxBlockCount; ++i)
    {
      minvar = min(minvar, minmax[i].x);
      maxvar = max(maxvar, minmax[i].y);
      //printf("%f %f\n",minmax[i].x,minmax[i].y);
    }
  
} // HydroWidgetGpu::computeMinMax

// =======================================================
// =======================================================
/** 
 * fill host array cmap_rgba.
 * The GPU version also copy this array to device memory to be used in
 * routine convertDataForPlotting
 * 
 */
void HydroWidgetGpu::initColormap()
{

  // call base class original version
  HydroWidget::initColormap();

  CUDA_SAFE_CALL( cudaMalloc((void **)&cmap_rgba_device, 
			     sizeof(unsigned int)*NBCOLORS));
  CUDA_SAFE_CALL( cudaMemcpy((void *)cmap_rgba_device,
			     (void *)cmap_rgba, sizeof(unsigned int)*NBCOLORS,
			     cudaMemcpyHostToDevice));

} // HydroWidgetGpu::initColormap

// =======================================================
// =======================================================
/** 
 * this is a wrapper to call the CUDA kernel which actually convert data
 * from h_U to the pixel buffer object.
 * @param _useColor : switch between colormap or greymap 
 */
void HydroWidgetGpu::convertDataForPlotting(int _useColor) {
  
  // first compute Min / Max values to properly handle contrast
  if (!manualContrast)
    computeMinMax(hydroRun->getData().data(),
		  hydroRun->getData().section(),
		  displayVar);
  
  dim3 grid = dim3(blocksFor(nx+4,PBO_BLOCK_DIMX), blocksFor(ny+4,PBO_BLOCK_DIMY));
  dim3 block = dim3(PBO_BLOCK_DIMX, PBO_BLOCK_DIMY);
  
  if (_useColor) {
    conversion_rgba_kernel<1><<<grid, block>>>(hydroRun->getData().data(), 
					       plot_rgba_pbo,
					       cmap_rgba_device,
					       NBCOLORS, 
					       hydroRun->getData().pitch(),
					       hydroRun->getData().dimx(), 
					       hydroRun->getData().dimy(), 
					       minvar, maxvar,
					       displayVar);  
  } else {
    conversion_rgba_kernel<0><<<grid, block>>>(hydroRun->getData().data(), 
					       plot_rgba_pbo,
					       cmap_rgba_device,
					       NBCOLORS, 
					       hydroRun->getData().pitch(),
					       hydroRun->getData().dimx(),
					       hydroRun->getData().dimy(),
					       minvar, maxvar,
					       displayVar);    
  }
  CUT_CHECK_ERROR("kernel conversion_rgba_kernel failed.");
  
} //HydroWidgetGpu::convertDataForPlotting 

// =======================================================
// =======================================================
void HydroWidgetGpu::createPBO()
{
  std::cout << "[DEBUG] GPU version of createPBO" << std::endl;

  // Create pixel buffer object and bind to gl_PBO. We store the data we want to
  // plot in memory on the graphics card - in a "pixel buffer". We can then 
  // copy this to the texture defined above and send it to the screen
  
  // Generate a buffer ID called a PBO (Pixel Buffer Object)
  GL_SAFE_CALL ( glGenBuffersARB(1, &gl_PBO) );
  
  // Make this the current UNPACK buffer (OpenGL is state-based)
  GL_SAFE_CALL ( glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, gl_PBO) );

  /*
   * CUDA only
   */
  // Copy PBO data to buffer
  //GL_SAFE_CALL ( glBufferDataARB(GL_PIXEL_UNPACK_BUFFER_ARB, hydroRun->getData().sectionBytes(), NULL, GL_STREAM_DRAW_ARB) );
  GL_SAFE_CALL ( glBufferDataARB(GL_PIXEL_UNPACK_BUFFER_ARB, hydroRun->getData().sectionBytes(), NULL, GL_STREAM_COPY_ARB) );
  //glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

#  ifdef USE_CUDA3
  CUDA_SAFE_CALL( cudaGraphicsGLRegisterBuffer( &cuda_PBO, gl_PBO, cudaGraphicsMapFlagsNone ) );
  cutilCheckMsg( "cudaGraphicsGLRegisterBuffer failed");
#  else
  CUDA_SAFE_CALL( cudaGLRegisterBufferObject(gl_PBO) );
  cutilCheckMsg( "cudaGLRegisterBufferObject failed");
#  endif // USE_CUDA3

  std::cout << "[DEBUG] PBO created." << std::endl;

} // HydroWidgetGpu::createPBO

// =======================================================
// =======================================================
void HydroWidgetGpu::deletePBO()
{
  if(gl_PBO) {
    // delete the gl_PBO

# ifdef USE_CUDA3
    CUDA_SAFE_CALL( cudaGraphicsUnregisterResource( cuda_PBO ) );
    cutilCheckMsg( "cudaGraphicsUnRegisterResource failed");
# else
    CUDA_SAFE_CALL( cudaGLUnregisterBufferObject( gl_PBO ) );
    cutilCheckMsg( "cudaGLUnRegisterBufferObject failed");
# endif // USE_CUDA3
    
    glDeleteBuffersARB( 1, &gl_PBO );
    gl_PBO=NULL;
    

# ifdef USE_CUDA3
    cuda_PBO = NULL;
# endif // USE_CUDA3

  }

  printf("PBO deleted...\n");

} // HydroWidgetGpu::deletePBO

// =======================================================
// =======================================================
void HydroWidgetGpu::render()
{
  // convert the plotvar array into an array of colors to plot

  CUDA_SAFE_THREAD_SYNC( );
  // For plotting, map the gl_PBO pixel buffer into CUDA context
  // space, so that CUDA can modify it
#ifdef USE_CUDA3
    CUDA_SAFE_CALL( cudaGraphicsMapResources(1, &cuda_PBO, NULL));
    cutilCheckMsg( "cudaGraphicsMapResources failed");

    size_t num_bytes; 
    CUDA_SAFE_CALL( cudaGraphicsResourceGetMappedPointer((void **)&plot_rgba_pbo, &num_bytes, cuda_PBO));
    cutilCheckMsg( "cudaGraphicsResourceGetMappedPointer failed");
#else
    CUDA_SAFE_CALL( cudaGLMapBufferObject((void**)&plot_rgba_pbo, gl_PBO) );
    cutilCheckMsg( "cudaGLMapBufferObject failed");
#endif // USE_CUDA3
  
  // Fill the plot_rgba_data array (and thus the pixel buffer)
  convertDataForPlotting(useColor);
  
  // unmap the PBO, so that OpenGL can safely do whatever he wants
#ifdef USE_CUDA3
  CUDA_SAFE_CALL( cudaGraphicsUnmapResources(1, &cuda_PBO, NULL));
  cutilCheckMsg( "cudaGraphicsUnmapResources failed" );
#else
  CUDA_SAFE_CALL( cudaGLUnmapBufferObject(gl_PBO) );
  cutilCheckMsg( "cudaGLUnmapBufferObject failed" );
#endif // USE_CUDA3

} // HydroWidgetGpu::render

// =======================================================
// =======================================================
void HydroWidgetGpu::initializeGL()
{
#ifdef USE_CUDA3
  printf("### using the new Cuda/OpenGL inter-operability API (Cuda >= 3.0)\n");
#else
  printf("### using the deprecated Cuda/OpenGL inter-operability API (Cuda < 3.0)\n");
#endif // USE_CUDA3

  HydroWidget::initializeGL();

  // #ifdef __CUDACC__
  //   CUDA_SAFE_CALL( cudaGLSetGLDevice( 0 ) );
  //   CUDA_SAFE_CALL( cudaGetLastError() );
  // #endif // __CUDACC__
  

} // HydroWidgetGpu::initializeGL
