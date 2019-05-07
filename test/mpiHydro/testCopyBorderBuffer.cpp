/**
 * \file testCopyBorderBuffer.cpp
 * \brief test routine copyBorderBuffer
  *
 * \date 14 Oct 2010
 * \author P. Kestener
 */

#include <cstdlib>
#include <iostream>

#include "hydro/mpiBorderUtils.h"

using namespace hydroSimu;

/*
 *
 */
int main(int argc, char * argv[]) {

  int isize = 8, jsize = 16;
  HostArray<real_t> U;
  HostArray<real_t> bx;
  HostArray<real_t> by;

  const int ghostWidth = 3;

  // memory allocation
  U.allocate ( make_uint3(isize     , jsize     , 1) );
  bx.allocate( make_uint3(ghostWidth, jsize     , 1) );
  by.allocate( make_uint3(isize     , ghostWidth, 1) );

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
  for (uint i=0; i<U.dimx(); ++i) {
    by(i,0,0) = -2.;
    by(i,1,0) = -2.;
    if (ghostWidth == 3)
      by(i,2,0) = -2.5;
  }

  std::cout << "Dump arrays:\n";
  std::cout << bx;
  std::cout << by;
  std::cout << U;

  // copy bx in the XMIN border
  copyBorderBufRecvToHostArray<XMIN,TWO_D,ghostWidth>(U,bx);
  std::cout << "Dump U after copy of bx in XMAX\n";
  std::cout << U;

  // copy by in the YMAX border
  copyBorderBufRecvToHostArray<YMAX,TWO_D,ghostWidth>(U,by);
  std::cout << "Dump U after copy of by in YMAX location\n";
  std::cout << U;


  
  return EXIT_SUCCESS;

}
