/**
 * \file glutGui/pbo.cuh
 * \brief Implement GPU kernels for converting real_t (float or double )array 
 * to scaled unsigned int for plotting purpose in the Pixel Buffer Object. 
 *
 * \date 24-02-2010
 * \author Pierre Kestener.
 *
 * $Id: pbo.cuh 1784 2012-02-21 10:34:58Z pkestene $
 */
#ifndef PBO_CUH_
#define PBO_CUH_

// for rbga conversion kernel
#define PBO_BLOCK_DIMX          16
#define PBO_BLOCK_DIMY          8


/** 
 * CUDA kernel to fill plot_rgba_data array for plotting
 * 
 * @param U 
 * @param plot_rgba_data 
 * @param cmap_rgba_data 
 * @param pitch 
 * @param minvar 
 * @param maxvar 
 * @param displayVar : 0 (density), 1 (Ux), 2 (Uy), 3 (Energy)
 */
template<int _useColor>
__global__ void conversion_rgba_kernel(real_t* U, 
				       unsigned int *plot_rgba_data, 
				       float *cmap,
				       float *cmap_der,
				       int ncol,
				       int pitch, 
				       int isize, int jsize,
				       real_t minvar, real_t maxvar,
				       int displayVar) 
{
  int i, j, i2d, i2dd;
  real_t frac;
  
  i = blockIdx.x*PBO_BLOCK_DIMX + threadIdx.x;
  j = blockIdx.y*PBO_BLOCK_DIMY + threadIdx.y;
  
  i2d  = __umul24(pitch, j) + i + displayVar*__umul24(pitch, jsize);;
  i2dd = __umul24(isize, j) + i;

  //frac = (U[i2d]-minvar)/(maxvar-minvar);
  frac = (maxvar-U[i2d])/(maxvar-minvar);
  if (frac<0)
    frac=0.0f;
  if (frac>1)
    frac=1.0f;

  frac *= 255.0;
  uint iCol = (uint) (frac);

  if (_useColor) {
    unsigned int r,g,b;
    r = (int) ( (cmap[3*iCol  ]  + (frac-iCol)*cmap_der[3*iCol  ] ) * 255.0 );
    g = (int) ( (cmap[3*iCol+1]  + (frac-iCol)*cmap_der[3*iCol+1] ) * 255.0 );
    b = (int) ( (cmap[3*iCol+2]  + (frac-iCol)*cmap_der[3*iCol+2] ) * 255.0 );
    
    plot_rgba_data[i2dd] = 
      (r << 24) | 
      (g << 16) |
      (b <<  8) |
      ((int)(frac) <<  0);
    
  } else {
    plot_rgba_data[i2dd] = 
      ((int)(255.0f) << 24) | // convert colourmap to int
      ((int)(frac) << 16) |
      ((int)(frac) <<  8) |
      ((int)(frac) <<  0);
  }
  
} //conversion_rgba_kernel

#endif // PBO_CUH_
