/**
 * \file testBorderBufferCuda.cpp
 * \brief test transfert border buffers (DeviceArray and HostArray classes).
 *
 * \date 25 Oct 2010
 * \author Pierre Kestener
 */
#include <mpi.h>

#include <cstdlib>
#include <iostream>
#include <typeinfo>

#include <common_types.h>
#include <constants.h>
#include <Arrays.h>
#include <mpiBorderUtils.h>

using namespace hydroSimu;

int main(int argc, char * argv[]) 
{
  
  try {
    
    // some test data
    //int isize = 150, jsize = 400;
    int isize = 200, jsize = 800;
    HostArray<real_t>    h_U;
    HostArray<real_t>    h_bTest;
    HostArray<real_t>    h_bTest2;
    DeviceArray<real_t>  d_U;
    DeviceArray<real_t>  d_bTemp;

    const int ghostWidth = 2;
    
    // memory allocation
    h_U.allocate     ( make_uint3(isize     , jsize, 1) );
    h_bTest.allocate ( make_uint3(ghostWidth, jsize, 1) );
    h_bTest2.allocate( make_uint3(ghostWidth, jsize, 1) );
    
    d_U.allocate     ( make_uint3(isize     , jsize, 1) );
    d_bTemp.allocate ( make_uint3(ghostWidth, jsize, 1) );
    
    // print info
    std::cout << "d_U.pitch()         " << d_U.pitch()         << std::endl;
    std::cout << "d_bTemp.pitch()     " << d_bTemp.pitch()     << std::endl;
    std::cout << "d_bTemp.dimx()      " << d_bTemp.dimx()      << std::endl;
    std::cout << "d_bTemp.dimXBytes() " << d_bTemp.dimXBytes() << std::endl;
    std::cout << "d_bTemp.dimy()      " << d_bTemp.dimy()      << std::endl;
    std::cout << "d_bTemp.dimz()      " << d_bTemp.dimz()      << std::endl;
 
    // initialization of host arrays
    for (uint j=0; j<h_U.dimy(); ++j)
      for (uint i=0; i<h_U.dimx(); ++i) {
	h_U(i,j,0) = static_cast<real_t>(i+j);
      }
    for (uint j=0; j<h_U.dimy(); ++j) {
      h_bTest(0,j,0) = -1.0;
      h_bTest(1,j,0) = -2.0;
    }
    
    // before buffer copy
    std::cout << "Print array U before border copy:\n\n";
    for (uint j=197; j<203; ++j) {
      std::cout << "[j: " << j << "] "; 
      for (uint i=0; i<5; ++i) {
	std::cout << h_U(i,j,0) << " ";
      }
      std::cout << "h_bTemp " << h_bTest(0,j,0) << " " <<  h_bTest(1,j,0);
      std::cout << std::endl;
    }
    
    // copy data to device arrays
    d_U.copyFromHost(h_U);
    //d_bTemp.copyFromHost(h_bTest);
    copyBorderBufRecvToDeviceArray<XMIN, TWO_D, ghostWidth>(d_U, d_bTemp, h_bTest);
    d_U.copyToHost(h_U);
    d_bTemp.copyToHost(h_bTest2);
        
    // after buffer copy
    std::cout << "Print array U after border copy:\n\n";
    for (uint j=197; j<203; ++j) {
      std::cout << "[j: " << j << "] "; 
      for (uint i=0; i<5; ++i) {
	std::cout << h_U(i,j,0) << " ";
      }
      std::cout << "h_bTemp " << h_bTest2(0,j,0) << " " <<  h_bTest2(1,j,0);
      std::cout << std::endl;
    }
    
    //copyBorderBufRecvToHostArray<XMAX,TWO_D>(h_U,h_bTest);
    
    std::cout << "#######################" << std::endl;
    std::cout << "Done !!!               " << std::endl;
    std::cout << "#######################" << std::endl;
          
  } catch (...) {
    
    std::cerr << "Exception caught, something really bad happened...\n\n\n";
    return EXIT_FAILURE;
    
  }
    
  return EXIT_SUCCESS;
  
}
