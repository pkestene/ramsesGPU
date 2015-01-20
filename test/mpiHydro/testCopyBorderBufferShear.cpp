/**
 * \file testCopyBorderBufferShear.cpp
 * \brief test routine copyBorderBufferShear
  *
 * \date 3 February 2012
 * \author P. Kestener
 */

#include <cstdlib>
#include <iostream>

//#include <Arrays.h>
#include <shearBorderUtils.h>

using namespace hydroSimu;

/*
 *
 */
int main(int argc, char * argv[]) {

  int isize = 8, jsize = 16;
  HostArray<real_t> U;
  HostArray<real_t> bx, bx_remap;

  const int ghostWidth = 3;

  // memory allocation
  U.allocate ( make_uint3(isize     , jsize     , 1) );
  bx.allocate( make_uint3(ghostWidth, jsize     , 1) );
  bx_remap.allocate( make_uint3(ghostWidth, jsize     ,1) );

  // initialization
  for (uint j=0; j<U.dimy(); ++j)
    for (uint i=0; i<U.dimx(); ++i) {
      U(i,j,0) = static_cast<real_t>(i+j);
    }
  for (uint j=0; j<U.dimy(); ++j) {
    bx(0,j,0) = -1.;
    bx(1,j,0) = -1.;
    if (ghostWidth == 3)
      bx(2,j,0) = -0.5;
  }
  for (uint j=0; j<U.dimy(); ++j) {
    bx_remap(0,j,0) = 111.;
    bx_remap(1,j,0) = 112.;
    if (ghostWidth == 3)
      bx_remap(2,j,0) = 113.;
  }
  std::cout << "Dump arrays:\n";
  std::cout << "bx\n" << bx;
  std::cout << "U\n"  << U;

  // copy xmin border of U into bx
  copyHostArrayToBorderBufShear<XMIN,TWO_D,ghostWidth>(bx,U);

  // copy bx into bx_remap
  std::cout << "########## Before remapping bx ############" << std::endl;
  std::cout << "bx      \n" << bx;
  std::cout << "bx_remap\n" << bx_remap;


  bx.copyTo(bx_remap);
  std::cout << "########## After  remapping bx ############" << std::endl;
  std::cout << "bx_remap\n"  << bx_remap;


  
  return EXIT_SUCCESS;

}
