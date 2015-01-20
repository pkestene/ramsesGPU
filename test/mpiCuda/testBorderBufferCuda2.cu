/**
 * \file testBorderBufferCuda2.cpp
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

#include "../../src/utils/monitoring/CudaTimer.h"
#include "../../src/utils/monitoring/measure_time.h"

using namespace hydroSimu;

int main(int argc, char * argv[]) 
{
  
  try {
    
    const int ghostWidth = 3;

    // some test data
    int isize = 2000, jsize=2000, ksize=4;
    HostArray<real_t>    h_border;
    HostArray<real_t>    h_borderPinned;
    DeviceArray<real_t>  d_border;
    DeviceArray<real_t>  d_U;
    
    // memory allocation
    h_border.allocate       ( make_uint3(ghostWidth, jsize, ksize), HostArray<real_t>::PAGEABLE );
    h_borderPinned.allocate ( make_uint3(ghostWidth, jsize, ksize), HostArray<real_t>::PINNED );
    d_border.allocate ( make_uint3(ghostWidth, jsize, ksize) );
    d_U.allocate      ( make_uint3(isize     , jsize, ksize) );
    
    // initialization of host arrays
    for (uint j=0; j<h_border.dimy(); ++j) {
      h_border(0,j,0) = -1.0;
      h_border(1,j,0) = -2.0;
      if (ghostWidth == 3)
	h_border(2,j,0) = -3.0;
    }
    
    // print
    std::cout << "###################################################" << std::endl;
    std::cout << " TEST 2D border with PITCHED memory on device      " << std::endl;
    std::cout << "                                                   " << std::endl;
    std::cout << "d_U      : " << d_U.dimy()      << "x" << d_U.dimy() << std::endl;
    std::cout << "h_border : " << h_border.dimx() << "x" << h_border.dimy() << "  pitch : " << d_border.pitch() << " size in bytes : " << h_border.sizeBytes() << std::endl; 
    std::cout << "d_border : " << d_border.dimx() << "x" << d_border.dimy() << "  pitch : " << d_border.pitch() << " size in bytes : " << h_border.sizeBytes() << std::endl; 
    std::cout << " isize : " << h_border.dimx() << std::endl;
    std::cout << " jsize : " << h_border.dimy() << std::endl;
    std::cout << " size in bytes (isize*2*NVAR_2D*sizeof(real_t)) : " << h_border.sizeBytes() << std::endl;
    CudaTimer timer;
    
    int nTransfert = 100;

    // host To Device (pageable memory)
    for (int i=0; i<nTransfert; ++i) {
      timer.start();
      copyBorderBufRecvToDeviceArray<XMIN, TWO_D, ghostWidth>(d_U, d_border, h_border);
      timer.stop();
    }
    std::cout << "transfert rate pageable (Host to Device): " << nTransfert*h_border.sizeBytes()/timer.elapsed()/1000000 << " MBytes/s" << std::endl;

    // Device to Host (pageable meory)
    timer.reset();
    for (int i=0; i<nTransfert; ++i) {
      timer.start();
      copyDeviceArrayToBorderBufSend<XMIN, TWO_D, ghostWidth>(h_border, d_border, d_U);
      timer.stop();
    }
    std::cout << "transfert rate pageable (Device to Host): " << nTransfert*h_border.sizeBytes()/timer.elapsed()/1000000 << " MBytes/s" << std::endl;

    // host To Device (pinned memory)
    timer.reset();
    for (int i=0; i<nTransfert; ++i) {
      timer.start();
      copyBorderBufRecvToDeviceArray<XMIN, TWO_D, ghostWidth>(d_U, d_border, h_borderPinned);
      timer.stop();
    }
    std::cout << "transfert rate pinned   (Host to Device): " << nTransfert*h_border.sizeBytes()/timer.elapsed()/1000000 << " MBytes/s" << std::endl;

    // Device to Host (pinned memory
    timer.reset();
    for (int i=0; i<nTransfert; ++i) {
      timer.start();
      copyDeviceArrayToBorderBufSend<XMIN, TWO_D, ghostWidth>(h_borderPinned, d_border, d_U);
      timer.stop();
    }
    std::cout << "transfert rate pinned   (Device to Host): " << nTransfert*h_border.sizeBytes()/timer.elapsed()/1000000 << " MBytes/s" << std::endl;

    {
      std::cout << std::endl;
      std::cout << "###################################################" << std::endl;
      std::cout << " TEST 3D border with LINEAR memory on device       " << std::endl;
      std::cout << "                                                   " << std::endl;
      isize = 200;
      jsize = 200;
      // direct test of cudaMalloc
      std::cout << "Direct test of cudaMalloc, cudaMemcpy and cudaMemcpyAsync" << std::endl;
      std::cout << "Use array size :  " << std::endl;
      std::cout << " isize : " << isize << std::endl;
      std::cout << " jsize : " << jsize << std::endl;
      
      size_t sizeInBytes = isize*jsize*2*5*sizeof(real_t);
      std::cout << " size in bytes (isize*jsize*2*NVAR_3D*sizeof(real_t)) : " << sizeInBytes << std::endl;

      real_t* d_data;
      cudaMalloc( (void **) &d_data , sizeInBytes);
      real_t* h_data;
      h_data = (real_t *) malloc(sizeInBytes);
      real_t* h_dataPinned;
      cudaMallocHost((void**)&h_dataPinned, sizeInBytes);

      // Host -> Device
      timer.reset();
      timer.start();
      for (int i=0; i<nTransfert; ++i) {
	cudaMemcpy((void *) d_data, h_data, sizeInBytes, cudaMemcpyHostToDevice);
      }
      timer.stop();
      std::cout << "transfert rate pageable (Host to Device): " << nTransfert*sizeInBytes/timer.elapsed()/1000000 << " MBytes/s" << std::endl;
      
      // Device -> Host
      timer.reset();
      timer.start();
      for (int i=0; i<nTransfert; ++i) {
	cudaMemcpy((void *) h_data, d_data, sizeInBytes, cudaMemcpyDeviceToHost);
      }
      timer.stop();
      std::cout << "transfert rate pageable (Device To Host): " << nTransfert*sizeInBytes/timer.elapsed()/1000000 << " MBytes/s" << std::endl;
      
      // Host -> Device async
      timer.reset();
      timer.start();
      for (int i=0; i<nTransfert; ++i) {
	cudaMemcpyAsync((void *) d_data, h_dataPinned, sizeInBytes, cudaMemcpyHostToDevice);
      }
      timer.stop();
      std::cout << "transfert rate pinned   (Host to Device): " << nTransfert*sizeInBytes/timer.elapsed()/1000000 << " MBytes/s" << std::endl;

      // Device -> Host async
      timer.reset();
      timer.start();
      for (int i=0; i<nTransfert; ++i) {
	cudaMemcpyAsync((void *) h_dataPinned, d_data, sizeInBytes, cudaMemcpyDeviceToHost);
      }
      timer.stop();
      std::cout << "transfert rate pinned   (Device To Host): " << nTransfert*sizeInBytes/timer.elapsed()/1000000 << " MBytes/s" << std::endl;

      cudaFree(d_data);
      free(h_data);
    }
    


  } catch (...) {
    
    std::cerr << "Exception caught, something really bad happened...\n\n\n";
    return EXIT_FAILURE;
    
  }
    
  return EXIT_SUCCESS;
  
}
