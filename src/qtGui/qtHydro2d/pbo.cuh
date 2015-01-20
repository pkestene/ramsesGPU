/**
 * \file qtHydro2d/pbo.cuh
 * \brief Implements GPU kernels for converting real_t (float or double )array 
 * to scaled unsigned int for plotting purpose in the Pixel Buffer Object. 
 *
 * \date 24-02-2010
 * \author Pierre Kestener.
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
				       unsigned int *cmap_rgba_data,
				       int ncol,
				       int pitch, 
				       int isize, int jsize,
				       real_t minvar, real_t maxvar,
				       int displayVar) 
{
  int i, j, i2d, i2dd/*, icol*/;
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

  //icol = (int)(frac * (real_t)ncol);
  if (_useColor) {
    //plot_rgba_data[i2dd] = cmap_rgba_data[icol];
    unsigned int r,g,b;
    r = FMIN( FMAX( 4*(frac-0.25), 0.), 1.);
    g = FMIN( FMAX( 4*FABS(frac-0.5)-1., 0.), 1.);
    b = FMIN( FMAX( 4*(0.75-frac), 0.), 1.);
    
    plot_rgba_data[i2dd] = ((int)(r*255.0f) << 24) | 
      ((int)(g * 255.0f) << 16) |
      ((int)(b * 255.0f) <<  8) |
      ((int)(frac*255.0f) <<  0);
    
  }else {
    plot_rgba_data[i2dd] = ((int)(255.0f) << 24) | // convert colourmap to int
      ((int)(frac * 255.0f) << 16) |
      ((int)(frac * 255.0f) <<  8) |
      ((int)(frac * 255.0f) <<  0);
  }
  
}

#endif // PBO_CUH_
