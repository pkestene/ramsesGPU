/*
 * Copyright CEA / Maison de la Simulation
 * Contributors: Pierre Kestener, Sebastien Fromang (May 22, 2012)
 *
 * This software is governed by the CeCILL license under French law and
 * abiding by the rules of distribution of free software.  You can  use, 
 * modify and/ or redistribute the software under the terms of the CeCILL
 * license as circulated by CEA, CNRS and INRIA at the following URL
 * "http://www.cecill.info". 
 *
 * The fact that you are presently reading this means that you have had
 * knowledge of the CeCILL license and that you accept its terms.
 */

/**
 * \file MHDRunBase.cpp
 * \brief Implements class MHDRunBase.
 *
 * \date March, 27 2011
 * \author P. Kestener
 *
 * $Id: MHDRunBase.cpp 3589 2014-11-02 23:02:09Z pkestene $
 */
#include "MHDRunBase.h"

// include CUDA kernel when necessary
#ifdef __CUDACC__
#include "cmpdt_mhd.cuh"
#include "resistivity.cuh"
#include "resistivity_zslab.cuh"
#include "mhd_ct_update.cuh"
#include "mhd_ct_update_zslab.cuh"
#endif // __CUDACC__

#include <limits> // for std::numeric_limits

#include "constoprim.h"
#include "mhd_utils.h"

#include "../utils/monitoring/date.h"
#include "ostream_fmt.h"
#include <cnpy.h>
#include "RandomGen.h"
#include "structureFunctions.h"

namespace hydroSimu {

  ////////////////////////////////////////////////////////////////////////////////
  // MHDRunBase class methods body
  ////////////////////////////////////////////////////////////////////////////////
  MHDRunBase::MHDRunBase(ConfigMap &_configMap) :
    HydroRunBase(_configMap)
    , history_method(NULL)
  {

    // time step computation (!!! GPU only !!!)
#ifdef __CUDACC__
    cmpdtBlockCount = std::min(cmpdtBlockCount, blocksFor(h_U.section(), CMPDT_BLOCK_SIZE * 2));

    h_invDt.allocate(make_uint3(cmpdtBlockCount, 1, 1));
    d_invDt.allocate(make_uint3(cmpdtBlockCount, 1, 1));
#endif // __CUDACC__

    // make sure variables declared as __constant__ are copied to device
    // for current compilation unit
    copyToSymbolMemory();

  }

  // =======================================================
  // =======================================================
  MHDRunBase::~MHDRunBase()
  {  
  } // MHDRunBase::~MHDRunBase

  // =======================================================
  // =======================================================
  real_t MHDRunBase::compute_dt_mhd(int useU)
#ifdef __CUDACC__
  {
    
    // choose between d_U and d_U2
    real_t *uData;
    if (useU == 0)
      uData = d_U.data();
    else
      uData = d_U2.data();

    // inverse time step
    real_t maxInvDt = 0;

    if (dimType == TWO_D) {

      cmpdt_2d_mhd<CMPDT_BLOCK_SIZE>
	<<<cmpdtBlockCount, 
	CMPDT_BLOCK_SIZE, 
	CMPDT_BLOCK_SIZE*sizeof(real_t)>>>(uData, 
					   d_invDt.data(), 
					   d_U.pitch(), 
					   d_U.dimx(),
					   d_U.dimy());
      checkCudaError("MHDRunBase cmpdt_2d_mhd error");
      d_invDt.copyToHost(h_invDt);
      checkCudaError("MHDRunBase h_invDt error");

    } else { // THREE_D

      cmpdt_3d_mhd<CMPDT_BLOCK_SIZE>
	<<<cmpdtBlockCount, 
	CMPDT_BLOCK_SIZE, 
	CMPDT_BLOCK_SIZE*sizeof(real_t)>>>(uData, 
					   d_invDt.data(), 
					   d_U.pitch(),
					   d_U.dimx(),
					   d_U.dimy(),
					   d_U.dimz());
      checkCudaError("MHDRunBase cmpdt_3d_mhd error");
      d_invDt.copyToHost(h_invDt);
      checkCudaError("MHDRunBase h_invDt error");
      
    } // end THREE_D

    real_t* invDt = h_invDt.data();
    
    for(uint i = 0; i < cmpdtBlockCount; ++i) {
      //std::cout << "invDt[" << i << "] = " << invDt[i] << std::endl;
      maxInvDt = FMAX ( maxInvDt, invDt[i]);
    }
    
    if (enableJet) {
      maxInvDt = FMAX ( maxInvDt, (this->ujet + this->cjet)/dx );
    }

    //std::cout << "computed dt : " << cfl/maxInvDt << std::endl;

    return cfl / maxInvDt;

  } // MHDRunBase::compute_dt_mhd

#else // CPU version of compute_dt_mhd

  {
    // time step inverse
    real_t invDt = gParams.smallc / FMIN(dx,dy);

    // choose between h_U and h_U2
    real_t *uData;
    if (useU == 0)
      uData = h_U.data();
    else
      uData = h_U2.data();

    /*
     * 
     */

    // section / domain size
    //int arraySize      =  (int) h_U.section();

    if (dimType == TWO_D) {
            
      int &geometry = ::gParams.geometry;

      int physicalDim[2] = {(int) h_U.pitch(), 
			    (int) h_U.dimy()};
      
      // for loop over inner region
      for (int j = ghostWidth; j < jsize-ghostWidth; j++)
	for (int i = ghostWidth; i < isize-ghostWidth; i++)
	  {
	    real_t q[NVAR_MHD];
	    real_t c;
	    int index = j*isize+i;
	    // convert conservative to primitive variables (stored in h_Q)
	    computePrimitives_MHD_2D(uData, physicalDim, index, c, q, ZERO_F);

	    // compute fastest information speeds
	    real_t fastInfoSpeed[3];
	    find_speed_info<TWO_D>(q, fastInfoSpeed);

	    real_t vx = fastInfoSpeed[IX];
	    real_t vy = fastInfoSpeed[IY];

	    if (enableJet) {
	      invDt = FMAX ( FMAX ( invDt, vx / dx + vy / dy ), (this->ujet + this->cjet)/dx );
	    } else {
	      invDt =        FMAX ( invDt, vx / dx + vy / dy );	      
	      if (geometry != GEO_CARTESIAN) {
		real_t xPos = ::gParams.xMin + dx/2 + (i-ghostWidth)*dx;
		invDt =      FMAX ( invDt, vx / dx + vy / dy / xPos );
	      }
	    }
 
	  } // end for(i,j)
            
    } else { // THREE_D

      real_t &Omega0   = ::gParams.Omega0;
      real_t  deltaX   = ::gParams.xMax - ::gParams.xMin;
      int    &geometry = ::gParams.geometry;

      int physicalDim[3] = {(int) h_U.pitch(),
			    (int) h_U.dimy(),
			    (int) h_U.dimz()};
      
      // for loop over inner region
      for (int k = ghostWidth; k < ksize-ghostWidth; k++)
	for (int j = ghostWidth; j < jsize-ghostWidth; j++)
	  for (int i = ghostWidth; i < isize-ghostWidth; i++)
	    {
	      real_t q[NVAR_MHD];
	      real_t c;
	      int index = k*jsize*isize+j*isize+i;
	      // convert conservative to primitive variables (stored in h_Q)
	      computePrimitives_MHD_3D(uData, physicalDim, index, c, q, ZERO_F);
	      
	      // compute fastest information speeds
	      real_t fastInfoSpeed[3];
	      find_speed_info<THREE_D>(q, fastInfoSpeed);

	      real_t vx = fastInfoSpeed[IX];
	      real_t vy = fastInfoSpeed[IY];
	      if (geometry == GEO_CARTESIAN and Omega0 > 0) {
		vy += 1.5*Omega0*deltaX/2;
	      }
	      real_t vz = fastInfoSpeed[IZ];

	      if (enableJet) {
		invDt = FMAX ( FMAX(  invDt, vx / dx + vy / dy + vz / dz ), (this->ujet + this->cjet)/dx );
	      } else {
		invDt =        FMAX ( invDt, vx / dx + vy / dy + vz / dz);
		/*if (invDt < 0.0001) 
		  std::cout << "problem with time step computation at (i,j,k) " <<i<<" "<<j<<" "<<k<<" ## "<< invDt << " " << 1/invDt << " " << vx/dx << " " << vy/dy << " " << vz/dz << std::endl;*/
	      }
	      
	      /* Special treatment for spherical geometry */
	      if (geometry == GEO_SPHERICAL)
		{
		  real_t xPos = ::gParams.xMin + dx/2 + (i-ghostWidth)*dx;
		  real_t yPos = ::gParams.yMin + dy/2 + (j-ghostWidth)*dy;
		  invDt =  FMAX ( invDt, vx / dx + vy / dy + vz / dz / xPos / sin(yPos));
		}
	      
	    } // end for(i,j,k)
      
    } // end THREE_D
    
    return cfl / invDt;

  } // MHDRunBase::compute_dt_mhd

#endif // __CUDACC__

  // =======================================================
  // =======================================================
  void MHDRunBase::compute_ct_update_2d(HostArray<real_t>  &U, 
					HostArray<real_t>  &emf,
					real_t dt)
  {
    
    real_t dtdx = dt/dx;
    real_t dtdy = dt/dy;

    // only update magnetic field
    for (int j=ghostWidth; j<jsize-ghostWidth+1; j++) {
      for (int i=ghostWidth; i<isize-ghostWidth+1; i++) {
	
	U(i  ,j  ,IA) += ( emf(i  ,j+1, I_EMFZ) - emf(i,j, I_EMFZ) )*dtdy;
	U(i  ,j  ,IB) -= ( emf(i+1,j  , I_EMFZ) - emf(i,j, I_EMFZ) )*dtdx;
	
      } // end for i
    } // end for j
    
  } // MHDRunBase::compute_ct_update_2d (2D case, CPU)
  
#ifdef __CUDACC__
  // =======================================================
  // =======================================================
  void MHDRunBase::compute_ct_update_2d(DeviceArray<real_t>  &U, 
					DeviceArray<real_t>  &emf,
					real_t dt)
  {
    
    dim3 dimBlock(MHD_CT_UPDATE_2D_DIMX,
		  MHD_CT_UPDATE_2D_DIMY);
    dim3 dimGrid(blocksFor(isize, MHD_CT_UPDATE_2D_DIMX),
		 blocksFor(jsize, MHD_CT_UPDATE_2D_DIMY));

    kernel_mhd_ct_update_2d<<< dimGrid, dimBlock >>> (U.data(), 
						      emf.data(),
						      U.pitch(),
						      U.dimx(), 
						      U.dimy(),
						      dt/dx, dt/dy, dt);
    checkCudaError("in MHDRunBase :: kernel_ct_update_2d");
    
  } // MHDRunBase::compute_ct_update_2d (2D case, GPU)
#endif // __CUDACC__

  // =======================================================
  // =======================================================
  void MHDRunBase::compute_ct_update_3d(HostArray<real_t>  &U, 
					HostArray<real_t>  &emf,
					real_t dt)
  {
    real_t dtdx = dt/dx;
    real_t dtdy = dt/dy;
    real_t dtdz = dt/dz;

    // only update magnetic field
    for (int k=ghostWidth; k<ksize-ghostWidth+1; k++) {
      for (int j=ghostWidth; j<jsize-ghostWidth+1; j++) {
	for (int i=ghostWidth; i<isize-ghostWidth+1; i++) {
	  
	  // update with EMFZ
	  if (k<ksize-ghostWidth) {

	    U(i ,j ,k, IA) += ( emf(i  ,j+1, k, I_EMFZ) - 
				emf(i,  j  , k, I_EMFZ) ) * dtdy;
	    
	    U(i ,j ,k, IB) -= ( emf(i+1,j  , k, I_EMFZ) - 
				emf(i  ,j  , k, I_EMFZ) ) * dtdx;
	    
	  }
	  
	  // update BX
	  U(i ,j ,k, IA) -= ( emf(i,j,k+1, I_EMFY) -
			      emf(i,j,k  , I_EMFY) ) * dtdz;
	  
	  // update BY
	  U(i ,j ,k, IB) += ( emf(i,j,k+1, I_EMFX) -
			      emf(i,j,k  , I_EMFX) ) * dtdz;
	  
	  // update BZ
	  U(i ,j ,k, IC) += ( emf(i+1,j  ,k, I_EMFY) -
			      emf(i  ,j  ,k, I_EMFY) ) * dtdx;
	  U(i ,j ,k, IC) -= ( emf(i  ,j+1,k, I_EMFX) -
			      emf(i  ,j  ,k, I_EMFX) ) * dtdy;
	  
	} // end for i
      } // end for j
    } // end for k
    
  } // MHDRunBase::compute_ct_update_3d (3D case, CPU)
  
#ifdef __CUDACC__
  // =======================================================
  // =======================================================
  void MHDRunBase::compute_ct_update_3d(DeviceArray<real_t>  &U, 
					DeviceArray<real_t>  &emf,
					real_t dt)
  {

    dim3 dimBlock(MHD_CT_UPDATE_3D_DIMX,
		  MHD_CT_UPDATE_3D_DIMY);
    dim3 dimGrid(blocksFor(isize, MHD_CT_UPDATE_3D_DIMX),
		 blocksFor(jsize, MHD_CT_UPDATE_3D_DIMY));
    
    kernel_mhd_ct_update_3d<<< dimGrid, dimBlock >>> (U.data(), 
						      emf.data(),
						      U.pitch(),
						      U.dimx(), U.dimy(), U.dimz(),
						      dt/dx, dt/dy, dt/dz, dt);
    checkCudaError("in MHDRunBase :: kernel_ct_update_3d");
    
  } // MHDRunBase::compute_ct_update_3d (3D case, GPU)
#endif // __CUDACC__

  // =======================================================
  // =======================================================
  void MHDRunBase::compute_ct_update_3d(HostArray<real_t>  &U, 
					HostArray<real_t>  &emf,
					real_t              dt,
					ZslabInfo           zSlabInfo)
  {
    // start and stop index of current slab (ghosts included)
    int& kStart = zSlabInfo.kStart;
    int& kStop  = zSlabInfo.kStop;

    real_t dtdx = dt/dx;
    real_t dtdy = dt/dy;
    real_t dtdz = dt/dz;

    int kStopUpdate = kStop-ghostWidth;
    if (zSlabInfo.zSlabId == zSlabInfo.zSlabNb-1)
      kStopUpdate = ksize-ghostWidth+1;

    // only update magnetic field
    for (int k = kStart+ghostWidth; k < kStopUpdate; k++) {

      // local k index
      int kL = k - kStart;
      	
	for (int j=ghostWidth; j<jsize-ghostWidth+1; j++) {
	  for (int i=ghostWidth; i<isize-ghostWidth+1; i++) {
	    
	    // update with EMFZ
	    if (k<ksize-ghostWidth) {

	      U(i ,j ,k, IA) += ( emf(i  ,j+1, kL, I_EMFZ) - 
				  emf(i,  j  , kL, I_EMFZ) ) * dtdy;
	      
	      U(i ,j ,k, IB) -= ( emf(i+1,j  , kL, I_EMFZ) - 
				  emf(i  ,j  , kL, I_EMFZ) ) * dtdx;

	    }
	    
	    // update BX
	    U(i ,j ,k, IA) -= ( emf(i,j,kL+1, I_EMFY) -
				emf(i,j,kL  , I_EMFY) ) * dtdz;
	    
	    // update BY
	    U(i ,j ,k, IB) += ( emf(i,j,kL+1, I_EMFX) -
				emf(i,j,kL  , I_EMFX) ) * dtdz;
	    
	    // update BZ
	    U(i ,j ,k, IC) += ( emf(i+1,j  ,kL, I_EMFY) -
				emf(i  ,j  ,kL, I_EMFY) ) * dtdx;
	    U(i ,j ,k, IC) -= ( emf(i  ,j+1,kL, I_EMFX) -
				emf(i  ,j  ,kL, I_EMFX) ) * dtdy;
	    
	  } // end for i
	} // end for j
	
    } // end for k
    
  } // MHDRunBase::compute_ct_update_3d (3D case, CPU, z-slab)
  
#ifdef __CUDACC__
  // =======================================================
  // =======================================================
  void MHDRunBase::compute_ct_update_3d(DeviceArray<real_t>  &U, 
					DeviceArray<real_t>  &emf,
					real_t dt,
					ZslabInfo zSlabInfo)
  {

    dim3 dimBlock(MHD_CT_UPDATE_Z_3D_DIMX,
		  MHD_CT_UPDATE_Z_3D_DIMY);
    dim3 dimGrid(blocksFor(isize, MHD_CT_UPDATE_Z_3D_DIMX),
		 blocksFor(jsize, MHD_CT_UPDATE_Z_3D_DIMY));
    
    kernel_mhd_ct_update_3d_zslab
      <<< dimGrid, dimBlock >>> (U.data(), 
				 emf.data(),
				 U.pitch(),
				 U.dimx(), U.dimy(), U.dimz(),
				 dt/dx, dt/dy, dt/dz, dt,
				 zSlabInfo);
    checkCudaError("in MHDRunBase :: kernel_ct_update_3d_zslab");
    
  } // MHDRunBase::compute_ct_update_3d (3D case, GPU, z-slab)
#endif // __CUDACC__

  // =======================================================
  // =======================================================
  void MHDRunBase::compute_resistivity_emf_2d(HostArray<real_t> &U,
					      HostArray<real_t> &emf)
  {
    real_t &eta  = _gParams.eta;

    real_t dbxdy = ZERO_F;
    //real_t dbxdz = ZERO_F;
    
    real_t dbydx = ZERO_F;
    //real_t dbydz = ZERO_F;
    
    //real_t dbzdx = ZERO_F;
    //real_t dbzdy = ZERO_F;
    
    //real_t jx    = ZERO_F;
    //real_t jy    = ZERO_F;
    real_t jz    = ZERO_F;
    
    // Compute J=curl(B)
    for (int j=ghostWidth; j<jsize-ghostWidth+1; j++) {
      for (int i=ghostWidth; i<isize-ghostWidth+1; i++) {
	
	dbydx = ( U(i,j,IBY) - U(i-1,j  ,IBY) ) / dx;
	//dbzdx = ( U(i,j,IBZ) - U(i-1,j  ,IBZ) ) / dx;
	
	dbxdy = ( U(i,j,IBX) - U(i  ,j-1,IBX) ) / dy;
	//dbzdy = ( U(i,j,IBZ) - U(i  ,j-1,IBZ) ) / dy;
	
	//dbxdz = ZERO_F;
	//dbydz = ZERO_F;
	
	//jx = dbzdy - dbydz;
	//jy = dbxdz - dbzdx;
	jz = dbydx - dbxdy;
	
	// note that multiplication by dt is done in ct
	emf(i,j,I_EMFZ) = -eta*jz;
	/*emf(i,j,I_EMFY) = -eta*jy;
	  emf(i,j,I_EMFX) = -eta*jx;*/
	
      } // end for i
    } // end for j

  } // MHDRunBase::compute_resistivity_emf (CPU, 2D)

#ifdef __CUDACC__
  // =======================================================
  // =======================================================
  void MHDRunBase::compute_resistivity_emf_2d(DeviceArray<real_t> &U,
					      DeviceArray<real_t> &emf)
  {
    dim3 dimBlock(RESISTIVITY_2D_DIMX,
		  RESISTIVITY_2D_DIMY);
    dim3 dimGrid(blocksFor(isize, RESISTIVITY_2D_DIMX),
		 blocksFor(jsize, RESISTIVITY_2D_DIMY));

    kernel_resistivity_forces_2d<<< dimGrid, 
      dimBlock >>> (U.data(), 
		    emf.data(),
		    ghostWidth, 
		    U.pitch(),
		    U.dimx(), U.dimy(), dx, dy);
    checkCudaError("in MHDRunBase :: kernel_resistivity_forces_2d");

  } // MHDRunBase::compute_resistivity_emf_2d (GPU)
#endif // __CUDACC__

  // =======================================================
  // =======================================================
  void MHDRunBase::compute_resistivity_emf_3d(HostArray<real_t> &U,
					      HostArray<real_t> &emf)
  {
    real_t &eta  = _gParams.eta;

    real_t dbxdy = ZERO_F;
    real_t dbxdz = ZERO_F;
    
    real_t dbydx = ZERO_F;
    real_t dbydz = ZERO_F;
    
    real_t dbzdx = ZERO_F;
    real_t dbzdy = ZERO_F;
    
    real_t jx=ZERO_F;
    real_t jy=ZERO_F;
    real_t jz=ZERO_F;
    
    // Compute J=curl(B)
    for (int k=ghostWidth; k<ksize-ghostWidth+1; k++) {
      for (int j=ghostWidth; j<jsize-ghostWidth+1; j++) {
	for (int i=ghostWidth; i<isize-ghostWidth+1; i++) {
	  
	  dbydx = ( U(i,j,k,IBY) - U(i-1,j  ,k  ,IBY) ) / dx;
	  dbzdx = ( U(i,j,k,IBZ) - U(i-1,j  ,k  ,IBZ) ) / dx;
	  
	  dbxdy = ( U(i,j,k,IBX) - U(i  ,j-1,k  ,IBX) ) / dy;
	  dbzdy = ( U(i,j,k,IBZ) - U(i  ,j-1,k  ,IBZ) ) / dy;
	  
	  dbxdz = ( U(i,j,k,IBX) - U(i  ,j  ,k-1,IBX) ) / dz;
	  dbydz = ( U(i,j,k,IBY) - U(i  ,j  ,k-1,IBY) ) / dz;
	  
	  jx = dbzdy - dbydz;
	  jy = dbxdz - dbzdx;
	  jz = dbydx - dbxdy;
	  
	  // note that multiplication by dt is done in ct
	  emf(i,j,k,I_EMFX) = -eta*jx;
	  emf(i,j,k,I_EMFY) = -eta*jy;
	  emf(i,j,k,I_EMFZ) = -eta*jz;
	  
	} // end for i
      } // end for j
    } // end for k

  } // MHDRunBase::compute_resistivity_emf (CPU, 3D)

#ifdef __CUDACC__
  // =======================================================
  // =======================================================
  void MHDRunBase::compute_resistivity_emf_3d(DeviceArray<real_t> &U,
					      DeviceArray<real_t> &emf)
  {
    dim3 dimBlock(RESISTIVITY_3D_DIMX,
		  RESISTIVITY_3D_DIMY);
    dim3 dimGrid(blocksFor(isize, RESISTIVITY_3D_DIMX),
		 blocksFor(jsize, RESISTIVITY_3D_DIMY));

    kernel_resistivity_forces_3d<<< dimGrid, 
      dimBlock >>> (U.data(), 
		    emf.data(),
		    ghostWidth, 
		    U.pitch(),
		    U.dimx(), U.dimy(), U.dimz(),
		    dx, dy, dz);
    checkCudaError("in MHDRunBase :: kernel_resistivity_forces_3d");

  } // MHDRunBase::compute_resistivity_emf_3d (GPU)
#endif // __CUDACC__

  // =======================================================
  // =======================================================
  void MHDRunBase::compute_resistivity_emf_3d(HostArray<real_t> &U,
					      HostArray<real_t> &emf,
					      ZslabInfo          zSlabInfo)
  {
    real_t &eta  = _gParams.eta;

    real_t dbxdy = ZERO_F;
    real_t dbxdz = ZERO_F;
    
    real_t dbydx = ZERO_F;
    real_t dbydz = ZERO_F;
    
    real_t dbzdx = ZERO_F;
    real_t dbzdy = ZERO_F;
    
    real_t jx=ZERO_F;
    real_t jy=ZERO_F;
    real_t jz=ZERO_F;
    
    // start and stop index of current slab (ghosts included)
    int& kStart = zSlabInfo.kStart;
    int& kStop  = zSlabInfo.kStop;
    
    // Compute J=curl(B)
    for (int k = kStart+ghostWidth; 
	 k     < kStop-ghostWidth+1; 
	 k++) {

      // local index inside slab
      int kL = k - kStart;

      if (k<ksize-ghostWidth+1) {
	
	for (int j=ghostWidth; j<jsize-ghostWidth+1; j++) {
	  for (int i=ghostWidth; i<isize-ghostWidth+1; i++) {
	    
	    dbydx = ( U(i,j,k,IBY) - U(i-1,j  ,k  ,IBY) ) / dx;
	    dbzdx = ( U(i,j,k,IBZ) - U(i-1,j  ,k  ,IBZ) ) / dx;
	    
	    dbxdy = ( U(i,j,k,IBX) - U(i  ,j-1,k  ,IBX) ) / dy;
	    dbzdy = ( U(i,j,k,IBZ) - U(i  ,j-1,k  ,IBZ) ) / dy;
	    
	    dbxdz = ( U(i,j,k,IBX) - U(i  ,j  ,k-1,IBX) ) / dz;
	    dbydz = ( U(i,j,k,IBY) - U(i  ,j  ,k-1,IBY) ) / dz;
	    
	    jx = dbzdy - dbydz;
	    jy = dbxdz - dbzdx;
	    jz = dbydx - dbxdy;
	    
	    // note that multiplication by dt is done in ct
	    emf(i,j,kL,I_EMFX) = -eta*jx;
	    emf(i,j,kL,I_EMFY) = -eta*jy;
	    emf(i,j,kL,I_EMFZ) = -eta*jz;
	    
	  } // end for i
	} // end for j

      } // end if (k<ksize-ghostWidth+1)

    } // end for k

  } // MHDRunBase::compute_resistivity_emf (CPU, 3D, z-slab)

#ifdef __CUDACC__
  // =======================================================
  // =======================================================
  void MHDRunBase::compute_resistivity_emf_3d(DeviceArray<real_t> &U,
					      DeviceArray<real_t> &emf,
					      ZslabInfo            zSlabInfo)
  {
    // take care that the last slab might be truncated
    if (zSlabInfo.zSlabId == zSlabInfo.zSlabNb-1) {
      zSlabInfo.ksizeSlab = ksize - zSlabInfo.kStart;
    }

    dim3 dimBlock(RESISTIVITY_Z_3D_DIMX,
		  RESISTIVITY_Z_3D_DIMY);
    dim3 dimGrid(blocksFor(isize, RESISTIVITY_Z_3D_DIMX),
		 blocksFor(jsize, RESISTIVITY_Z_3D_DIMY));

    kernel_resistivity_forces_3d_zslab<<< dimGrid, 
      dimBlock >>> (U.data(), 
		    emf.data(),
		    ghostWidth, 
		    U.pitch(),
		    U.dimx(), U.dimy(), U.dimz(),
		    dx, dy, dz,
		    zSlabInfo);
    checkCudaError("in MHDRunBase :: kernel_resistivity_forces_3d_zslab");

  } // MHDRunBase::compute_resistivity_emf_3d (GPU)
#endif // __CUDACC__

  // =======================================================
  // =======================================================
  void MHDRunBase::compute_resistivity_energy_flux_2d(HostArray<real_t> &U,
						      HostArray<real_t> &fluxX,
						      HostArray<real_t> &fluxY,
						      real_t             dt)
  {

    real_t bx,   by,   bz;
    real_t jx,   jy,   jz;
    real_t /*jxp1, jyp1,*/ jzp1;

    real_t &eta  = _gParams.eta;

    for (int j=ghostWidth; j<jsize-ghostWidth+1; j++) {
      for (int i=ghostWidth; i<isize-ghostWidth+1; i++) {
	
	// 1st direction energy flux

	by = ( U(i,j  ,IBY) + U(i-1,j  ,IBY) + 
	       U(i,j+1,IBY) + U(i-1,j+1,IBY) )/4;
	
	bz = ( U(i  ,j,IBZ) + 
	       U(i-1,j,IBZ) )/2;
	  
	jy   = - ( U(i  ,j  ,IBZ) - 
		   U(i-1,j  ,IBZ) )/dx;
	
	jz   = 
	  ( U(i,j  ,IBY) - U(i-1,j  ,IBY) )/dx -
	  ( U(i,j  ,IBX) - U(i  ,j-1,IBX) )/dy;
	jzp1 = 
	  ( U(i,j+1,IBY) - U(i-1,j+1,IBY) )/dx -
	  ( U(i,j+1,IBX) - U(i  ,j  ,IBX) )/dy;
	jz   = (jz+jzp1)/2;
	
	fluxX(i,j,ID) = ZERO_F;
	fluxX(i,j,IP) = - eta*(jy*bz-jz*by)*dt/dx;
	fluxX(i,j,IU) = ZERO_F;
	fluxX(i,j,IV) = ZERO_F;

	// 2nd direction energy flux
	
	bx = ( U(i  ,j,IBX) + U(i  ,j-1,IBX) +
	       U(i+1,j,IBX) + U(i+1,j-1,IBX) )/4;

	bz = ( U(i  ,j,IBZ) + U(i  ,j-1,IBZ) )/2;
	
	jx = ( U(i  ,j  ,IBZ) -
	       U(i  ,j-1,IBZ) )/dy;
	
	jz   = 
	  ( U(i  ,j,IBY) - U(i-1,j  ,IBY) )/dx -
	  ( U(i  ,j,IBX) - U(i  ,j-1,IBX) )/dy;

	jzp1 = 
	  ( U(i+1,j,IBY) - U(i  ,j  ,IBY) )/dx -
	  ( U(i+1,j,IBX) - U(i+1,j-1,IBX) )/dy;

	jz = (jz+jzp1)/2;
          
	fluxY(i,j,ID) = ZERO_F;
	fluxY(i,j,IP) = - eta*(jz*bx-jx*bz)*dt/dy;
	fluxY(i,j,IU) = ZERO_F;
	fluxY(i,j,IV) = ZERO_F;	
	  	  
	} // end for i
      } // end for j

  } // MHDRunBase::compute_resistivity_energy_flux_2d
  
  // =======================================================
  // =======================================================
#ifdef __CUDACC__
  void MHDRunBase::compute_resistivity_energy_flux_2d(DeviceArray<real_t> &U,
						      DeviceArray<real_t> &fluxX,
						      DeviceArray<real_t> &fluxY,
						      real_t               dt)
  {

    dim3 dimBlock(RESISTIVITY_ENERGY_2D_DIMX,
		  RESISTIVITY_ENERGY_2D_DIMY);
    dim3 dimGrid(blocksFor(isize, RESISTIVITY_ENERGY_2D_DIMX),
		 blocksFor(jsize, RESISTIVITY_ENERGY_2D_DIMY));

    kernel_resistivity_energy_flux_2d<<< dimGrid, 
      dimBlock >>> (U.data(), 
		    fluxX.data(),
		    fluxY.data(),
		    ghostWidth, 
		    U.pitch(),
		    U.dimx(), U.dimy(), dx, dy, dt);
    checkCudaError("in MHDRunBase :: kernel_resistivity_energy_flux_2d");

  } // MHDRunBase::compute_resistivity_energy_flux_2d (GPU)
#endif // __CUDACC

  // =======================================================
  // =======================================================
  void MHDRunBase::compute_resistivity_energy_flux_3d(HostArray<real_t> &U,
						      HostArray<real_t> &fluxX,
						      HostArray<real_t> &fluxY,
						      HostArray<real_t> &fluxZ,
						      real_t             dt)
  {

    real_t bx,   by,   bz;
    real_t jx,   jy,   jz;
    real_t jxp1, jyp1, jzp1;

    real_t &eta  = _gParams.eta;

    for (int k=ghostWidth; k<ksize-ghostWidth+1; k++) {
      for (int j=ghostWidth; j<jsize-ghostWidth+1; j++) {
	for (int i=ghostWidth; i<isize-ghostWidth+1; i++) {
           
	  // 1st direction energy flux

	  by = ( U(i,j  ,k,IBY) + U(i-1,j  ,k,IBY) + 
		 U(i,j+1,k,IBY) + U(i-1,j+1,k,IBY) )/4;
	  bz = ( U(i,j,k  ,IBZ) + U(i-1,j,k  ,IBZ) + 
		 U(i,j,k+1,IBZ) + U(i-1,j,k+1,IBZ) )/4;
	  
	  jy   = 
	    ( U(i,j,k  ,IBX) - U(i  ,j  ,k-1,IBX) )/dz -
	    ( U(i,j,k  ,IBZ) - U(i-1,j  ,k  ,IBZ) )/dx;
	  jyp1 = 
	    ( U(i,j,k+1,IBX) - U(i  ,j  ,k  ,IBX) )/dz -
	    ( U(i,j,k+1,IBZ) - U(i-1,j  ,k+1,IBZ) )/dx;
	  jy   = (jy+jyp1)/2;
	  
	  jz   = 
	    ( U(i,j  ,k,IBY) - U(i-1,j  ,k  ,IBY) )/dx -
	    ( U(i,j  ,k,IBX) - U(i  ,j-1,k  ,IBX) )/dy;
	  jzp1 = 
	    ( U(i,j+1,k,IBY) - U(i-1,j+1,k  ,IBY) )/dx -
	    ( U(i,j+1,k,IBX) - U(i  ,j  ,k  ,IBX) )/dy;
	  jz   = (jz+jzp1)/2;
	  
	  fluxX(i,j,k,ID) = ZERO_F;
	  fluxX(i,j,k,IP) = - eta*(jy*bz-jz*by)*dt/dx;
	  fluxX(i,j,k,IU) = ZERO_F;
	  fluxX(i,j,k,IV) = ZERO_F;
	  fluxX(i,j,k,IW) = ZERO_F;

	  // 2nd direction energy flux
	  
	  bx = ( U(i  ,j,k,IBX) + U(i  ,j-1,k,IBX) +
		 U(i+1,j,k,IBX) + U(i+1,j-1,k,IBX) )/4;
	  
	  
	  bz = ( U(i,j,k  ,IBZ) + U(i,j-1,k  ,IBZ) + 
		 U(i,j,k+1,IBZ) + U(i,j-1,k+1,IBZ) )/4;
	  
	  jx   = 
	    ( U(i,j,k  ,IBZ) - U(i  ,j-1,k  ,IBZ) )/dy -
	    ( U(i,j,k  ,IBY) - U(i  ,j  ,k-1,IBY) )/dz;
	  jxp1 = 
	    ( U(i,j,k+1,IBZ) - U(i  ,j-1,k+1,IBZ) )/dy -
	    ( U(i,j,k+1,IBY) - U(i  ,j  ,k  ,IBY) )/dz;
	  jx = (jx+jxp1)/2;
	    
	  
	  jz   = 
	    ( U(i  ,j,k,IBY) - U(i-1,j  ,k  ,IBY) )/dx -
	    ( U(i  ,j,k,IBX) - U(i  ,j-1,k  ,IBX) )/dy;
	  jzp1 = 
	    ( U(i+1,j,k,IBY) - U(i  ,j  ,k  ,IBY) )/dx -
	    ( U(i+1,j,k,IBX) - U(i+1,j-1,k  ,IBX) )/dy;
	  jz = (jz+jzp1)/2;
	  
	  fluxY(i,j,k,ID) = ZERO_F;
	  fluxY(i,j,k,IP) = - eta*(jz*bx-jx*bz)*dt/dy;
	  fluxY(i,j,k,IU) = ZERO_F;
	  fluxY(i,j,k,IV) = ZERO_F;
	  fluxY(i,j,k,IW) = ZERO_F;
	  
	  // 3rd direction energy flux
	  bx = ( U(i  ,j,k,IBX) + U(i  ,j,k-1,IBX) + 
		 U(i+1,j,k,IBX) + U(i+1,j,k-1,IBX) )/4;
	  by = ( U(i,j  ,k,IBY) + U(i,j  ,k-1,IBY) + 
		 U(i,j+1,k,IBY) + U(i,j+1,k-1,IBY) )/4;
	  
	  jx   = 
	    ( U(i,j  ,k,IBZ) - U(i  ,j-1,k  ,IBZ) )/dy -
	    ( U(i,j  ,k,IBY) - U(i  ,j  ,k-1,IBY) )/dz;
	  jxp1 = 
	    ( U(i,j+1,k,IBZ) - U(i  ,j  ,k  ,IBZ) )/dy -
	    ( U(i,j+1,k,IBY) - U(i  ,j+1,k-1,IBY) )/dz;
	  jx   = (jx+jxp1)/2;

	  jy   = 
	    ( U(i  ,j,k,IBX) - U(i  ,j  ,k-1,IBX) )/dz -
	    ( U(i  ,j,k,IBZ) - U(i-1,j  ,k  ,IBZ) )/dx;
	  jyp1 = 
	    ( U(i+1,j,k,IBX) - U(i+1,j  ,k-1,IBX) )/dz -
	    ( U(i+1,j,k,IBZ) - U(i  ,j  ,k  ,IBZ) )/dx;
	  jy   = (jy+jyp1)/2;
          
	  fluxZ(i,j,k,ID) = ZERO_F;
	  fluxZ(i,j,k,IP) = - eta*(jx*by-jy*bx)*dt/dz;
	  fluxZ(i,j,k,IU) = ZERO_F;
	  fluxZ(i,j,k,IV) = ZERO_F;
	  fluxZ(i,j,k,IW) = ZERO_F;
	  
	} // end for i
      } // end for j
    } // end for k

  } // MHDRunBase::compute_resistivity_energy_flux_3d
  
  // =======================================================
  // =======================================================
#ifdef __CUDACC__
  void MHDRunBase::compute_resistivity_energy_flux_3d(DeviceArray<real_t> &U,
						      DeviceArray<real_t> &fluxX,
						      DeviceArray<real_t> &fluxY,
						      DeviceArray<real_t> &fluxZ,
						      real_t               dt)
  {

    dim3 dimBlock(RESISTIVITY_ENERGY_3D_DIMX,
		  RESISTIVITY_ENERGY_3D_DIMY);
    dim3 dimGrid(blocksFor(isize, RESISTIVITY_ENERGY_3D_DIMX),
		 blocksFor(jsize, RESISTIVITY_ENERGY_3D_DIMY));

    kernel_resistivity_energy_flux_3d<<< dimGrid, 
      dimBlock >>> (U.data(), 
		    fluxX.data(),
		    fluxY.data(),
		    fluxZ.data(),
		    ghostWidth, 
		    U.pitch(),
		    U.dimx(), U.dimy(), U.dimz(),
		    dx, dy, dz, dt);
    checkCudaError("in MHDRunBase :: kernel_resistivity_energy_flux_3d");

  } // MHDRunBase::compute_resistivity_energy_flux_3d
#endif // __CUDACC

  // =======================================================
  // =======================================================
  void MHDRunBase::compute_resistivity_energy_flux_3d(HostArray<real_t> &U,
						      HostArray<real_t> &fluxX,
						      HostArray<real_t> &fluxY,
						      HostArray<real_t> &fluxZ,
						      real_t             dt,
						      ZslabInfo          zSlabInfo)
  {

    real_t bx,   by,   bz;
    real_t jx,   jy,   jz;
    real_t jxp1, jyp1, jzp1;
    
    real_t &eta  = _gParams.eta;
    
    // start and stop index of current slab (ghosts included)
    int& kStart = zSlabInfo.kStart;
    int& kStop  = zSlabInfo.kStop;
    
    for (int k = kStart+ghostWidth; 
	 k     < kStop-ghostWidth+1; 
	 k++) {

      // local index inside slab
      int kL = k - kStart;
      
      if (k<ksize-ghostWidth+1) {
	
	for (int j=ghostWidth; j<jsize-ghostWidth+1; j++) {
	  for (int i=ghostWidth; i<isize-ghostWidth+1; i++) {
	    
	    // 1st direction energy flux
	    
	    by = ( U(i,j  ,k,IBY) + U(i-1,j  ,k,IBY) + 
		   U(i,j+1,k,IBY) + U(i-1,j+1,k,IBY) )/4;
	    bz = ( U(i,j,k  ,IBZ) + U(i-1,j,k  ,IBZ) + 
		   U(i,j,k+1,IBZ) + U(i-1,j,k+1,IBZ) )/4;
	    
	    jy   = 
	      ( U(i,j,k  ,IBX) - U(i  ,j  ,k-1,IBX) )/dz -
	      ( U(i,j,k  ,IBZ) - U(i-1,j  ,k  ,IBZ) )/dx;
	    jyp1 = 
	      ( U(i,j,k+1,IBX) - U(i  ,j  ,k  ,IBX) )/dz -
	      ( U(i,j,k+1,IBZ) - U(i-1,j  ,k+1,IBZ) )/dx;
	    jy   = (jy+jyp1)/2;
	    
	    jz   = 
	      ( U(i,j  ,k,IBY) - U(i-1,j  ,k  ,IBY) )/dx -
	      ( U(i,j  ,k,IBX) - U(i  ,j-1,k  ,IBX) )/dy;
	    jzp1 = 
	      ( U(i,j+1,k,IBY) - U(i-1,j+1,k  ,IBY) )/dx -
	      ( U(i,j+1,k,IBX) - U(i  ,j  ,k  ,IBX) )/dy;
	    jz   = (jz+jzp1)/2;
	    
	    fluxX(i,j,kL,ID) = ZERO_F;
	    fluxX(i,j,kL,IP) = - eta*(jy*bz-jz*by)*dt/dx;
	    fluxX(i,j,kL,IU) = ZERO_F;
	    fluxX(i,j,kL,IV) = ZERO_F;
	    fluxX(i,j,kL,IW) = ZERO_F;
	    
	    // 2nd direction energy flux
	    
	    bx = ( U(i  ,j,k,IBX) + U(i  ,j-1,k,IBX) +
		   U(i+1,j,k,IBX) + U(i+1,j-1,k,IBX) )/4;
	    
	    
	    bz = ( U(i,j,k  ,IBZ) + U(i,j-1,k  ,IBZ) + 
		   U(i,j,k+1,IBZ) + U(i,j-1,k+1,IBZ) )/4;
	    
	    jx   = 
	      ( U(i,j,k  ,IBZ) - U(i  ,j-1,k  ,IBZ) )/dy -
	      ( U(i,j,k  ,IBY) - U(i  ,j  ,k-1,IBY) )/dz;
	    jxp1 = 
	      ( U(i,j,k+1,IBZ) - U(i  ,j-1,k+1,IBZ) )/dy -
	      ( U(i,j,k+1,IBY) - U(i  ,j  ,k  ,IBY) )/dz;
	    jx = (jx+jxp1)/2;
	    
	    
	    jz   = 
	      ( U(i  ,j,k,IBY) - U(i-1,j  ,k  ,IBY) )/dx -
	      ( U(i  ,j,k,IBX) - U(i  ,j-1,k  ,IBX) )/dy;
	    jzp1 = 
	      ( U(i+1,j,k,IBY) - U(i  ,j  ,k  ,IBY) )/dx -
	      ( U(i+1,j,k,IBX) - U(i+1,j-1,k  ,IBX) )/dy;
	    jz = (jz+jzp1)/2;
	    
	    fluxY(i,j,kL,ID) = ZERO_F;
	    fluxY(i,j,kL,IP) = - eta*(jz*bx-jx*bz)*dt/dy;
	    fluxY(i,j,kL,IU) = ZERO_F;
	    fluxY(i,j,kL,IV) = ZERO_F;
	    fluxY(i,j,kL,IW) = ZERO_F;
	    
	    // 3rd direction energy flux
	    bx = ( U(i  ,j,k,IBX) + U(i  ,j,k-1,IBX) + 
		   U(i+1,j,k,IBX) + U(i+1,j,k-1,IBX) )/4;
	    by = ( U(i,j  ,k,IBY) + U(i,j  ,k-1,IBY) + 
		   U(i,j+1,k,IBY) + U(i,j+1,k-1,IBY) )/4;
	    
	    jx   = 
	      ( U(i,j  ,k,IBZ) - U(i  ,j-1,k  ,IBZ) )/dy -
	      ( U(i,j  ,k,IBY) - U(i  ,j  ,k-1,IBY) )/dz;
	    jxp1 = 
	      ( U(i,j+1,k,IBZ) - U(i  ,j  ,k  ,IBZ) )/dy -
	      ( U(i,j+1,k,IBY) - U(i  ,j+1,k-1,IBY) )/dz;
	    jx   = (jx+jxp1)/2;
	    
	    jy   = 
	      ( U(i  ,j,k,IBX) - U(i  ,j  ,k-1,IBX) )/dz -
	      ( U(i  ,j,k,IBZ) - U(i-1,j  ,k  ,IBZ) )/dx;
	    jyp1 = 
	      ( U(i+1,j,k,IBX) - U(i+1,j  ,k-1,IBX) )/dz -
	      ( U(i+1,j,k,IBZ) - U(i  ,j  ,k  ,IBZ) )/dx;
	    jy   = (jy+jyp1)/2;
	    
	    fluxZ(i,j,kL,ID) = ZERO_F;
	    fluxZ(i,j,kL,IP) = - eta*(jx*by-jy*bx)*dt/dz;
	    fluxZ(i,j,kL,IU) = ZERO_F;
	    fluxZ(i,j,kL,IV) = ZERO_F;
	    fluxZ(i,j,kL,IW) = ZERO_F;
	    
	  } // end for i
	} // end for j

      } // end if (k<ksize-ghostWidth+1)

    } // end for k

  } // MHDRunBase::compute_resistivity_energy_flux_3d (z-slab)
  
  // =======================================================
  // =======================================================
#ifdef __CUDACC__
  void MHDRunBase::compute_resistivity_energy_flux_3d(DeviceArray<real_t> &U,
						      DeviceArray<real_t> &fluxX,
						      DeviceArray<real_t> &fluxY,
						      DeviceArray<real_t> &fluxZ,
						      real_t               dt,
						      ZslabInfo            zSlabInfo)
  {

    // take care that the last slab might be truncated
    if (zSlabInfo.zSlabId == zSlabInfo.zSlabNb-1) {
      zSlabInfo.ksizeSlab = ksize - zSlabInfo.kStart;
    }

    dim3 dimBlock(RESISTIVITY_ENERGY_Z_3D_DIMX,
		  RESISTIVITY_ENERGY_Z_3D_DIMY);
    dim3 dimGrid(blocksFor(isize, RESISTIVITY_ENERGY_Z_3D_DIMX),
		 blocksFor(jsize, RESISTIVITY_ENERGY_Z_3D_DIMY));

    kernel_resistivity_energy_flux_3d_zslab<<< dimGrid, 
      dimBlock >>> (U.data(), 
		    fluxX.data(),
		    fluxY.data(),
		    fluxZ.data(),
		    ghostWidth, 
		    U.pitch(),
		    U.dimx(), U.dimy(), U.dimz(),
		    dx, dy, dz, dt,
		    zSlabInfo);
    checkCudaError("in MHDRunBase :: kernel_resistivity_energy_flux_3d_zslab");

  } // MHDRunBase::compute_resistivity_energy_flux_3d (z-slab)
#endif // __CUDACC

  // =======================================================
  // =======================================================
  void MHDRunBase::compute_divB(HostArray<real_t>& h_conserv, 
				HostArray<real_t>& h_divB) 
  {
    /*
     * compute magnetic field divergence
     */
    
    if (dimType == TWO_D) {
      
      for (int j=0; j<jsize-1; j++) {
	
	for (int i=0; i<isize-1; i++) {
	  
	  h_divB(i,j,0) = 
	    (h_conserv(i+1,j  ,IA)-h_conserv(i,j,IA))/dx +
	    (h_conserv(i  ,j+1,IB)-h_conserv(i,j,IB))/dy;

	} // end for i

      } // end for j
      
    } else { // THREE_D

      for (int k=0; k<ksize-1; k++) {
	
	for (int j=0; j<jsize-1; j++) {
	  
	  for (int i=0; i<isize-1; i++) {
	    
	    h_divB(i,j,k,0) = 
	      (h_conserv(i+1,j  ,k  ,IA)-h_conserv(i,j,k,IA))/dx +
	      (h_conserv(i  ,j+1,k  ,IB)-h_conserv(i,j,k,IB))/dy +
	      (h_conserv(i  ,j  ,k+1,IC)-h_conserv(i,j,k,IC))/dz;

	  } // end for i
	  
	} // end for j
	
      } // end for k
      
    } // end THREE_D
    
  } // MHDRunBase::compute_divB
  
  // =======================================================
  // =======================================================
  void MHDRunBase::compute_laplacianB(HostArray<real_t>& h_conserv, 
				      HostArray<real_t>& h_laplacianB) 
  {
    
    // put zeros everywhere
    h_laplacianB.reset();
    
    /*
     * compute magnetic field laplacian
     */
    if (dimType == TWO_D) {
      
      for (int j=1; j<jsize-1; j++) {
	
	for (int i=1; i<isize-1; i++) {
	  
	  h_laplacianB(i,j,IX) = 
	    (h_conserv(i+1,j  ,IA)-h_conserv(i,j,IA))/dx +
	    (h_conserv(i-1,j  ,IA)-h_conserv(i,j,IA))/dx +
	    (h_conserv(i  ,j+1,IA)-h_conserv(i,j,IA))/dx +
	    (h_conserv(i  ,j-1,IA)-h_conserv(i,j,IA))/dx;
	  
	  h_laplacianB(i,j,IY) = 
	    (h_conserv(i+1,j  ,IB)-h_conserv(i,j,IB))/dy +
	    (h_conserv(i-1,j  ,IB)-h_conserv(i,j,IB))/dy +
	    (h_conserv(i  ,j+1,IB)-h_conserv(i,j,IB))/dy +
	    (h_conserv(i  ,j-1,IB)-h_conserv(i,j,IB))/dy;
	  
	} // end for i

      } // end for j
      
    } else { // THREE_D

      for (int k=1; k<ksize-1; k++) {
	
	for (int j=1; j<jsize-1; j++) {
	  
	  for (int i=1; i<isize-1; i++) {

	    real_t normB = sqrt( SQR(h_conserv(i,j,k,IA)) +
				 SQR(h_conserv(i,j,k,IB)) +
				 SQR(h_conserv(i,j,k,IC)) );
	    
	    h_laplacianB(i,j,k,IX) = 
	      (h_conserv(i+1,j  ,k  ,IA)-h_conserv(i,j,k,IA)) +
	      (h_conserv(i-1,j  ,k  ,IA)-h_conserv(i,j,k,IA)) +
	      (h_conserv(i  ,j+1,k  ,IA)-h_conserv(i,j,k,IA)) +
	      (h_conserv(i  ,j-1,k  ,IA)-h_conserv(i,j,k,IA)) +
	      (h_conserv(i  ,j  ,k+1,IA)-h_conserv(i,j,k,IA)) +
	      (h_conserv(i  ,j  ,k-1,IA)-h_conserv(i,j,k,IA));
	    h_laplacianB(i,j,k,IX) /= normB;
	    //h_laplacianB(i,j,k,IX) /= dx;
	  
	    h_laplacianB(i,j,k,IY) = 
	      (h_conserv(i+1,j  ,k  ,IB)-h_conserv(i,j,k,IB)) +
	      (h_conserv(i-1,j  ,k  ,IB)-h_conserv(i,j,k,IB)) +
	      (h_conserv(i  ,j+1,k  ,IB)-h_conserv(i,j,k,IB)) +
	      (h_conserv(i  ,j-1,k  ,IB)-h_conserv(i,j,k,IB)) +
	      (h_conserv(i  ,j  ,k+1,IB)-h_conserv(i,j,k,IB)) +
	      (h_conserv(i  ,j  ,k-1,IB)-h_conserv(i,j,k,IB));
	    h_laplacianB(i,j,k,IY) /= normB;
	    //h_laplacianB(i,j,k,IY) /= dy; 
	    
	    h_laplacianB(i,j,k,IZ) = 
	      (h_conserv(i+1,j  ,k  ,IC)-h_conserv(i,j,k,IC)) +
	      (h_conserv(i-1,j  ,k  ,IC)-h_conserv(i,j,k,IC)) +
	      (h_conserv(i  ,j+1,k  ,IC)-h_conserv(i,j,k,IC)) +
	      (h_conserv(i  ,j-1,k  ,IC)-h_conserv(i,j,k,IC)) +
	      (h_conserv(i  ,j  ,k+1,IC)-h_conserv(i,j,k,IC)) +
	      (h_conserv(i  ,j  ,k-1,IC)-h_conserv(i,j,k,IC));
	    h_laplacianB(i,j,k,IZ) /= normB;
	    //h_laplacianB(i,j,k,IZ) /=  dz;

	  } // end for i
	  
	} // end for j
	
      } // end for k
      
    } // end THREE_D
    
  } // MHDRunBase::compute_divB
  
  // =======================================================
  // =======================================================
  int MHDRunBase::init_simulation(const std::string problemName)
  {
    
    // test if we are performing a re-start run (default : false)
    bool restartEnabled = configMap.getBool("run","restart",false);
    int timeStep = 0;

    if (restartEnabled) { // load data from input data file

      /* initial condition in grid interior */
      memset(h_U.data(),0,h_U.sizeBytes());
     
      // get input filename from configMap
      std::string inputFilename = configMap.getString("run", "restart_filename", "");
      
      // get output directory
      std::string outputDir    = configMap.getString("output", "outputDir", "./");

      // upscale init data from a file twice smaller
      bool restartUpscaleEnabled = configMap.getBool("run","restart_upscale",false);
      
      if (restartUpscaleEnabled) { // load low resolution data from file
      
	// allocate h_input (half resolution, ghost included)
	HostArray<real_t> h_input;
	h_input.allocate(make_uint4(nx/2+2*ghostWidth, 
				    ny/2+2*ghostWidth,
				    nz/2+2*ghostWidth,
				    nbVar));

	// read input date into temporary array h_input
	bool halfResolution=true;
	inputHdf5(h_input, outputDir+"/"+inputFilename, halfResolution);

	// upscale h_input into h_U (i.e. double resolution)
	upscale(h_U, h_input);

      } else { // standard restart

	// read input HDF5 file into h_U buffer , and return time Step
	timeStep = inputHdf5(h_U, outputDir+"/"+inputFilename);

      } // if (restartUpscaleEnabled)

      // in case of turbulence problem, we also need to re-initialize the
      // random forcing field
      if (!problemName.compare("turbulence")) {
	this->init_randomForcing();
      }

      // some extra stuff that need to be done here
      restart_run_extra_work();

    } else { // perform regular initialization

      if (!problemName.compare("Orszag-Tang") || 
	  !problemName.compare("OrszagTang") ) {
	this->init_Orszag_Tang();
      } else if (!problemName.compare("jet") ||
		 !problemName.compare("Jet")) {
	this->init_mhd_jet();
      } else if (!problemName.compare("sod")) {
	this->init_mhd_sod();
      } else if (!problemName.compare("Brio-Wu") || 
		 !problemName.compare("BrioWu")  || 
		 !problemName.compare("brio-wu") || 
		 !problemName.compare("briowu")) {
	this->init_mhd_BrioWu();
      } else if (!problemName.compare("Rotor") || 
		 !problemName.compare("rotor")) {
	this->init_mhd_rotor();
      } else if (!problemName.compare("FieldLoop")  ||
		 !problemName.compare("fieldloop")  ||
		 !problemName.compare("Fieldloop")  ||
		 !problemName.compare("field-loop") || 
		 !problemName.compare("Field-Loop")) {
	this->init_mhd_field_loop();
      } else if (!problemName.compare("CurrentSheet") ||
		 !problemName.compare("currentsheet") ||
		 !problemName.compare("Current-Sheet") ||
		 !problemName.compare("current-sheet") ||
		 !problemName.compare("Currentsheet")) {
	this->init_mhd_current_sheet();
      } else if (!problemName.compare("InertialWave") ||
		 !problemName.compare("inertialwave") ||
		 !problemName.compare("Inertial-Wave") ||
		 !problemName.compare("inertial-wave") ||
		 !problemName.compare("Inertialwave")) {
	this->init_mhd_inertial_wave();
      } else if (!problemName.compare("ShearWave") ||
		 !problemName.compare("shearwave") ||
		 !problemName.compare("Shear-Wave") ||
		 !problemName.compare("shear-wave") ||
		 !problemName.compare("Shearwave")) {
	this->init_mhd_shear_wave();
      } else if (!problemName.compare("MRI") ||
		 !problemName.compare("Mri") ||
		 !problemName.compare("mri")) {
	this->init_mhd_mri();
      } else if (!problemName.compare("Kelvin-Helmholtz")) {
	this->init_mhd_Kelvin_Helmholtz();
      } else if (!problemName.compare("Rayleigh-Taylor")) {
	this->init_mhd_Rayleigh_Taylor();
      } else if (!problemName.compare("turbulence")) {
	this->init_mhd_turbulence();
      } else if (!problemName.compare("turbulence-Ornstein-Uhlenbeck")) {
	this->init_mhd_turbulence_Ornstein_Uhlenbeck();
      } else {
	std::cerr << "given problem parameter is: " << problem << std::endl;
	std::cerr << "unknown problem name; please set MHD parameter \"problem\" to a valid value !!!" << std::endl;
	
      }

    } // end regular initialization

    // copy data to GPU if necessary
#ifdef __CUDACC__
    d_U.reset();
    d_U.copyFromHost(h_U); // load data into the VRAM
    d_U2.reset();
    d_U2.copyFromHost(h_U); // load data into the VRAM
#else
    // copy h_U into h_U2
    h_U.copyTo(h_U2);
#endif // __CUDACC__

    // do we force timeStep to be zero ?
    bool resetTimeStep = configMap.getBool("run","restart_reset_timestep",false);
    if (resetTimeStep)
      timeStep=0;

    return timeStep;

  } // MHDRunBase::init_simulation
  
  // =======================================================
  // =======================================================
  /**
   * Orszag-Tang Vortex problem.
   *
   * adapted from Dumses/patch/ot/condinit.f90 original fortran code.
   * The energy initialization asserts periodic boundary conditions.
   *
   *
   * \sa http://www.astro.virginia.edu/VITA/ATHENA/ot.html
   */
  void MHDRunBase::init_Orszag_Tang()
  {

    if (!mhdEnabled) {
      std::cerr << "MHD must be enabled to use this initial conditions !!!";
      return;
    }

    /* first set zero everywhere */
    memset(h_U.data(),0,h_U.sizeBytes());

    const double TwoPi = 4.0*asin(1.0);
    const double B0    = 1.0/sqrt(2.0*TwoPi);
    const double p0    = (double) (_gParams.gamma0/(2.0*TwoPi));
    const double d0    = (double) (_gParams.gamma0*p0);
    const double v0    = 1.0;

    real_t &xMin = _gParams.xMin;
    real_t &yMin = _gParams.yMin;
    real_t &zMin = _gParams.zMin;

    real_t &xMax = _gParams.xMax;
    real_t &yMax = _gParams.yMax;
    real_t &zMax = _gParams.zMax;

    if (dimType == TWO_D) {

      for (int j=ghostWidth*0; j<jsize-ghostWidth*0; j++) {

	double yPos = yMin + dy/2 + (j-ghostWidth)*dy;

	for (int i=ghostWidth*0; i<isize-ghostWidth*0; i++) {

	  double xPos = xMin + dx/2 + (i-ghostWidth)*dx;

	  // density initialization
	  h_U(i,j,ID)  = static_cast<real_t>(d0);
          
	  // rho*vx
	  h_U(i,j,IU)  = static_cast<real_t>(-d0*v0*sin(yPos*TwoPi));
	  
	  // rho*vy
	  h_U(i,j,IV)  = static_cast<real_t>( d0*v0*sin(xPos*TwoPi));
	  
	  // rho*vz
	  h_U(i,j,IW) =  ZERO_F;
          
	  // bx
	  h_U(i,j,IBX) = static_cast<real_t>(-B0*sin(    yPos*TwoPi));
          
	  // by
	  h_U(i,j,IBY) = static_cast<real_t>( B0*sin(2.0*xPos*TwoPi));
	  
	  // bz
	  h_U(i,j,IBZ) = ZERO_F;
	  
	} // end for i

      } // end for j

      // total energy (periodic boundary conditions taken into account)
      // see original fortran code in Dumses/patch/ot/condinit.f90
      for (int j=0; j<jsize; j++) {
	
	for (int i=0; i<isize; i++) {
	  
	  if (i<isize-1 and j<jsize-1) {
	    h_U(i,j,IP)  = p0 / (_gParams.gamma0-1.0) +
	      0.5 * ( SQR(h_U(i,j,IU)) / h_U(i,j,ID) +
		      SQR(h_U(i,j,IV)) / h_U(i,j,ID) +
		      0.25*SQR(h_U(i,j,IBX) + h_U(i+1,j  ,IBX)) + 
		      0.25*SQR(h_U(i,j,IBY) + h_U(i  ,j+1,IBY)) );
	  } else if ( (i <isize-1) and (j==jsize-1)) {
	    h_U(i,j,IP)  = p0 / (_gParams.gamma0-1.0) +
	      0.5 * ( SQR(h_U(i,j,IU)) / h_U(i,j,ID) +
		      SQR(h_U(i,j,IV)) / h_U(i,j,ID) +
		      0.25*SQR(h_U(i,j,IBX) + h_U(i+1,j           ,IBX)) + 
		      0.25*SQR(h_U(i,j,IBY) + h_U(i  ,2*ghostWidth,IBY)) );
	  } else if ( (i==isize-1) and (j <jsize-1)) {
	    h_U(i,j,IP)  = p0 / (_gParams.gamma0-1.0) +
	      0.5 * ( SQR(h_U(i,j,IU)) / h_U(i,j,ID) +
		      SQR(h_U(i,j,IV)) / h_U(i,j,ID) +
		      0.25*SQR(h_U(i,j,IBX) + h_U(2*ghostWidth,j  ,IBX)) + 
		      0.25*SQR(h_U(i,j,IBY) + h_U(i           ,j+1,IBY)) );
	  } else if ( (i==isize-1) and (j==jsize-1) ) {
	    h_U(i,j,IP)  = p0 / (_gParams.gamma0-1.0) +
	      0.5 * ( SQR(h_U(i,j,IU)) / h_U(i,j,ID) +
		      SQR(h_U(i,j,IV)) / h_U(i,j,ID) +
		      0.25*SQR(h_U(i,j,IBX) + h_U(2*ghostWidth,j           ,IBX)) + 
		      0.25*SQR(h_U(i,j,IBY) + h_U(i           ,2*ghostWidth,IBY)) );
	  }
	  

	} // end for i
	
      }	// end for j

    } else { // THREE_D

      /* get direction (only usefull for 3D) : 0->XY, 1->YZ, 2->ZX */
      int direction = configMap.getInteger("OrszagTang","direction",0);
      if (direction < 0 || direction > 3) {
	direction = 0;
	std::cout << "Orszag-Tang direction set to X-Y plane" << std::endl;
      }

      // transverse wave number (3D only)
      const double kt = configMap.getFloat("OrszagTang","kt",0.0);
      
      if (direction == 0) { // vortex in X-Y plane

	for (int k=0; k<ksize; k++) {
	
	  double zPos = zMin + dz/2 + (k-ghostWidth)*dz;

	  for (int j=0; j<jsize; j++) {
	  
	    double yPos = yMin + dy/2 + (j-ghostWidth)*dy;
	  
	    for (int i=0; i<isize; i++) {
	    
	      double xPos = xMin + dx/2 + (i-ghostWidth)*dx;
	    
	      // density initialization
	      h_U(i,j,k,ID)  = static_cast<real_t>(d0);
	      
	      // rho*vx
	      h_U(i,j,k,IU)  = static_cast<real_t>(- d0*v0*sin(yPos*TwoPi));
	      
	      // rho*vy
	      h_U(i,j,k,IV)  = static_cast<real_t>(  d0*v0*sin(xPos*TwoPi));
	      
	      // rho*vz
	      h_U(i,j,k,IW) =  ZERO_F;
	      
	      // bx
	      h_U(i,j,k,IBX) = static_cast<real_t>(-B0*cos(2*TwoPi*kt*(zPos-zMin)/(zMax-zMin))*sin(    yPos*TwoPi));

	      // by
	      h_U(i,j,k,IBY) = static_cast<real_t>( B0*cos(2*TwoPi*kt*(zPos-zMin)/(zMax-zMin))*sin(2.0*xPos*TwoPi));
	      
	      // bz
	      h_U(i,j,k,IBZ) = ZERO_F;
	      
	    } // end for i
	    
	  } // end for j
	  
	} // end for k
	
	// total energy (periodic boundary conditions taken into account)
	// see original fortran code in Dumses/patch/ot/condinit.f90
	for (int k=0; k<ksize; k++) {
	  
	  for (int j=0; j<jsize; j++) {
	    
	    for (int i=0; i<isize; i++) {
	      
	      if (i<isize-1 and j<jsize-1) {
		h_U(i,j,k,IP)  = p0 / (_gParams.gamma0-1.0) +
		  0.5 * ( SQR(h_U(i,j,k,IU)) / h_U(i,j,k,ID) +
			  SQR(h_U(i,j,k,IV)) / h_U(i,j,k,ID) +
			  0.25*SQR(h_U(i,j,k,IBX) + h_U(i+1,j  ,k,IBX)) + 
			  0.25*SQR(h_U(i,j,k,IBY) + h_U(i  ,j+1,k,IBY)) );
	      } else if ( (i <isize-1) and (j==jsize-1)) {
		h_U(i,j,IP)  = p0 / (_gParams.gamma0-1.0) +
		0.5 * ( SQR(h_U(i,j,k,IU)) / h_U(i,j,k,ID) +
			SQR(h_U(i,j,k,IV)) / h_U(i,j,k,ID) +
			0.25*SQR(h_U(i,j,k,IBX) + h_U(i+1,j           ,k,IBX)) + 
			0.25*SQR(h_U(i,j,k,IBY) + h_U(i  ,2*ghostWidth,k,IBY)) );
	      } else if ( (i==isize-1) and (j <jsize-1)) {
		h_U(i,j,IP)  = p0 / (_gParams.gamma0-1.0) +
		  0.5 * ( SQR(h_U(i,j,k,IU)) / h_U(i,j,k,ID) +
			  SQR(h_U(i,j,k,IV)) / h_U(i,j,k,ID) +
			  0.25*SQR(h_U(i,j,k,IBX) + h_U(2*ghostWidth,j  ,k,IBX)) + 
			  0.25*SQR(h_U(i,j,k,IBY) + h_U(i           ,j+1,k,IBY)) );
	      } else if ( (i==isize-1) and (j==jsize-1) ) {
		h_U(i,j,IP)  = p0 / (_gParams.gamma0-1.0) +
		  0.5 * ( SQR(h_U(i,j,k,IU)) / h_U(i,j,k,ID) +
			  SQR(h_U(i,j,k,IV)) / h_U(i,j,k,ID) +
			  0.25*SQR(h_U(i,j,k,IBX) + h_U(2*ghostWidth,j           ,k,IBX)) + 
			  0.25*SQR(h_U(i,j,k,IBY) + h_U(i           ,2*ghostWidth,k,IBY)) );
	      }
	      
	      
	    } // end for i
	    
	  } // end for j
	  
	} // end for k
	
      } else if (direction == 1) { // vortex in plane YZ
	
	for (int i=0; i<isize; i++) {
	  
	  double xPos = xMin + dx/2 + (i-ghostWidth)*dx;

	  for (int k=0; k<ksize; k++) {
	    
	    double zPos = zMin + dz/2 + (k-ghostWidth)*dz;
	    
	    for (int j=0; j<jsize; j++) {
	      
	      double yPos = yMin + dy/2 + (j-ghostWidth)*dy;
	      
	      // density initialization
	      h_U(i,j,k,ID)  = static_cast<real_t>(d0);
	      
	      // rho*vy
	      h_U(i,j,k,IV)  = static_cast<real_t>(- d0*v0*sin(zPos*TwoPi));
	      
	      // rho*vz
	      h_U(i,j,k,IW)  = static_cast<real_t>(  d0*v0*sin(yPos*TwoPi));
	      
	      // rho*vx
	      h_U(i,j,k,IU) = ZERO_F;
	      
	      // by
	      h_U(i,j,k,IBY) = static_cast<real_t>(-B0*cos(2*TwoPi*kt*(xPos-xMin)/(xMax-xMin))*sin(    zPos*TwoPi));
	      
	      // bz
	      h_U(i,j,k,IBZ) = static_cast<real_t>( B0*cos(2*TwoPi*kt*(xPos-xMin)/(xMax-xMin))*sin(2.0*yPos*TwoPi));
	      
	      // bx
	      h_U(i,j,k,IBX) = ZERO_F;
	      
	    } // end for j
	    
	  } // end for k
	  
	} // end for i
	
	// total energy (periodic boundary conditions taken into account)
	// see original fortran code in Dumses/patch/ot/condinit.f90
	for (int i=0; i<isize; i++) {
	  
	  for (int k=0; k<ksize; k++) {
	    
	    for (int j=0; j<jsize; j++) {
	      
	      if (j<jsize-1 and k<ksize-1) {
		h_U(i,j,k,IP)  = p0 / (_gParams.gamma0-1.0) +
		  0.5 * ( SQR(h_U(i,j,k,IV)) / h_U(i,j,k,ID) +
			  SQR(h_U(i,j,k,IW)) / h_U(i,j,k,ID) +
			  0.25*SQR(h_U(i,j,k,IBY) + h_U(i,j+1,k  ,IBY)) + 
			  0.25*SQR(h_U(i,j,k,IBZ) + h_U(i,j  ,k+1,IBZ)) );
	      } else if ( (j<jsize-1) and (k==ksize-1)) {
		h_U(i,j,IP)  = p0 / (_gParams.gamma0-1.0) +
		  0.5 * ( SQR(h_U(i,j,k,IV)) / h_U(i,j,k,ID) +
			  SQR(h_U(i,j,k,IW)) / h_U(i,j,k,ID) +
			  0.25*SQR(h_U(i,j,k,IBY) + h_U(i,j+1,k           ,IBY)) +
			  0.25*SQR(h_U(i,j,k,IBZ) + h_U(i,j  ,2*ghostWidth,IBZ)) );
	      } else if ( (j==jsize-1) and (k<ksize-1)) {
		h_U(i,j,IP)  = p0 / (_gParams.gamma0-1.0) +
		  0.5 * ( SQR(h_U(i,j,k,IV)) / h_U(i,j,k,ID) +
			  SQR(h_U(i,j,k,IW)) / h_U(i,j,k,ID) +
			  0.25*SQR(h_U(i,j,k,IBY) + h_U(i,2*ghostWidth,k  ,IBY)) + 
			  0.25*SQR(h_U(i,j,k,IBZ) + h_U(i,j           ,k+1,IBZ)) );
	      } else if ( (j==jsize-1) and (k==ksize-1) ) {
		h_U(i,j,IP)  = p0 / (_gParams.gamma0-1.0) +
		  0.5 * ( SQR(h_U(i,j,k,IV)) / h_U(i,j,k,ID) +
			  SQR(h_U(i,j,k,IW)) / h_U(i,j,k,ID) +
			  0.25*SQR(h_U(i,j,k,IBY) + h_U(i,2*ghostWidth,k,IBY)) + 
			  0.25*SQR(h_U(i,j,k,IBZ) + h_U(i,j,2*ghostWidth,IBZ)) );
	      }
	      
	    } // end for j
	    
	  } // end for k
	  
	} // end for i
	
      } else if (direction == 2) { // vortex in direction ZX

	for (int j=0; j<jsize; j++) {
	  
	  double yPos = yMin + dy/2 + (j-ghostWidth)*dy;
	  
	  for (int i=0; i<isize; i++) {
	    
	    double xPos = xMin + dx/2 + (i-ghostWidth)*dx;
	    
	    for (int k=0; k<ksize; k++) {
	      
	      double zPos = zMin + dz/2 + (k-ghostWidth)*dz;
	      
	      // density initialization
	      h_U(i,j,k,ID)  = static_cast<real_t>(d0);
	      
	      // rho*vz
	      h_U(i,j,k,IW)  = static_cast<real_t>(-d0*v0*sin(xPos*TwoPi));
	      
	      // rho*vx
	      h_U(i,j,k,IU)  = static_cast<real_t>( d0*v0*sin(zPos*TwoPi));
	      
	      // rho*vy
	      h_U(i,j,k,IV) =  ZERO_F;
	      
	      // bz
	      h_U(i,j,k,IBZ) = static_cast<real_t>(-B0*cos(2*TwoPi*kt*(yPos-yMin)/(yMax-yMin))*sin(    xPos*TwoPi));
	      
	      // bx
	      h_U(i,j,k,IBX) = static_cast<real_t>( B0*cos(2*TwoPi*kt*(yPos-yMin)/(yMax-yMin))*sin(2.0*zPos*TwoPi));
	      
	      // by
	      h_U(i,j,k,IBY) = ZERO_F;
	      
	    } // end for k
	    
	  } // end for x
	  
	} // end for j
	
	// total energy (periodic boundary conditions taken into account)
	// see original fortran code in Dumses/patch/ot/condinit.f90
	for (int j=0; j<jsize; j++) {
	  
	  for (int i=0; i<isize; i++) {
	    
	    for (int k=0; k<ksize; k++) {
	      
	      if (k<ksize-1 and i<isize-1) {
		h_U(i,j,k,IP)  = p0 / (_gParams.gamma0-1.0) +
		  0.5 * ( SQR(h_U(i,j,k,IW)) / h_U(i,j,k,ID) +
			  SQR(h_U(i,j,k,IU)) / h_U(i,j,k,ID) +
			  0.25*SQR(h_U(i,j,k,IBZ) + h_U(i  ,j,k+1,IBZ)) + 
			  0.25*SQR(h_U(i,j,k,IBX) + h_U(i+1,j,k  ,IBX)) );
	      } else if ( (k<ksize-1) and (i==isize-1) ) {
		h_U(i,j,IP)  = p0 / (_gParams.gamma0-1.0) +
		  0.5 * ( SQR(h_U(i,j,k,IW)) / h_U(i,j,k,ID) +
			  SQR(h_U(i,j,k,IU)) / h_U(i,j,k,ID) +
			  0.25*SQR(h_U(i,j,k,IBZ) + h_U(i           ,j,k+1,IBZ)) +
			  0.25*SQR(h_U(i,j,k,IBX) + h_U(2*ghostWidth,j,k  ,IBX)) );
	      } else if ( (k==ksize-1) and (i<isize-1) ) {
		h_U(i,j,IP)  = p0 / (_gParams.gamma0-1.0) +
		  0.5 * ( SQR(h_U(i,j,k,IW)) / h_U(i,j,k,ID) +
			  SQR(h_U(i,j,k,IU)) / h_U(i,j,k,ID) +
			  0.25*SQR(h_U(i,j,k,IBZ) + h_U(i  ,j,2*ghostWidth,IBZ)) + 
			  0.25*SQR(h_U(i,j,k,IBX) + h_U(i+1,j,k           ,IBX)) );
	      } else if ( (k==ksize-1) and (i==isize-1) ) {
		h_U(i,j,IP)  = p0 / (_gParams.gamma0-1.0) +
		  0.5 * ( SQR(h_U(i,j,k,IW)) / h_U(i,j,k,ID) +
			  SQR(h_U(i,j,k,IU)) / h_U(i,j,k,ID) +
			  0.25*SQR(h_U(i,j,k,IBZ) + h_U(i,j,2*ghostWidth,IBZ)) + 
			  0.25*SQR(h_U(i,j,k,IBX) + h_U(2*ghostWidth,j,k,IBX)) );
	      }
	      
	    } // end for j
	    
	  } // end for k
	  
	} // end for i

      } // end direction
      
    } // end THREE_D

  } // MHDRunBase::init_Ortzag_Tang

  // =======================================================
  // =======================================================
  /**
   * Same as in hydro (just for testing that without magnetic field,
   * everything is working correctly).
   * This a pure hydro test ! But you can have a static magnetic field !
   *
   */
  void MHDRunBase::init_mhd_jet()
  {    

    if (!mhdEnabled) {
      std::cerr << "MHD must be enabled to use this initial conditions !!!";
      return;
    }

    /* read Static magnetic field component */
    real_t Bx = configMap.getFloat("jet","BStatic_x",0.0);
    real_t By = configMap.getFloat("jet","BStatic_y",0.0);
    real_t Bz = configMap.getFloat("jet","BStatic_z",0.0);

    /* initial condition in grid interior */
    memset(h_U.data(),0,h_U.sizeBytes());
    
    if (dimType == TWO_D) {

      /* jet */
      for (int j=ghostWidth; j<jsize-ghostWidth; j++)
	for (int i=ghostWidth; i<isize-ghostWidth; i++) {

	  h_U(i,j,ID)=1.0f;
	  h_U(i,j,IP)=1.0f/(_gParams.gamma0-1.0f) + 0.5*(Bx*Bx+By*By);
	  h_U(i,j,IU)=0.0f;
	  h_U(i,j,IV)=0.0f;
	  h_U(i,j,IW)=0.0f;
	  h_U(i,j,IA)=Bx;
	  h_U(i,j,IB)=By;
	  h_U(i,j,IC)=Bz;
	}

    } else { // THREE_D

      /* jet */
      for (int k=ghostWidth; k<ksize-ghostWidth; k++)
	for (int j=ghostWidth; j<jsize-ghostWidth; j++)
	  for (int i=ghostWidth; i<isize-ghostWidth; i++) {
	    
	    h_U(i,j,k,ID)=1.0f;
	    h_U(i,j,k,IP)=1.0f/(_gParams.gamma0-1.0f) + 0.5*(Bx*Bx+By*By+Bz*Bz);
	    h_U(i,j,k,IU)=0.0f;
	    h_U(i,j,k,IV)=0.0f;
	    h_U(i,j,k,IW)=0.0f;
	    h_U(i,j,k,IA)=Bx;
	    h_U(i,j,k,IB)=By;
	    h_U(i,j,k,IC)=Bz;
	  }

    } // end THREE_D

  } // MHDRunBase::init_mhd_jet

  // =======================================================
  // =======================================================
  /**
   * Sod test.
   *
   */
  void MHDRunBase::init_mhd_sod()
  {    

    if (!mhdEnabled) {
      std::cerr << "MHD must be enabled to use this initial conditions !!!";
      return;
    }

    /* initial condition in grid interior */
    memset(h_U.data(),0,h_U.sizeBytes());
    
    if (dimType == TWO_D) {

      /* Sod test */
      for (int j=ghostWidth; j<jsize-ghostWidth; j++)
	for (int i=ghostWidth; i<isize-ghostWidth; i++) {

	  if (i < isize/2) {
	    h_U(i,j,ID) = 1.0f;
	    h_U(i,j,IP) = 1.0f/(_gParams.gamma0-1.0f);
	  } else {
	    h_U(i,j,ID) = 0.125f;
	    h_U(i,j,IP) = 0.1f/(_gParams.gamma0-1.0f);
	  }
	  h_U(i,j,IU)=0.0f;
	  h_U(i,j,IV)=0.0f;
	  h_U(i,j,IW)=0.0f;
	  h_U(i,j,IA)=0.0f;
	  h_U(i,j,IB)=0.0f;
	  h_U(i,j,IC)=0.0f;
	}

    } else { // THREE_D
    
      /* Sod test */
      for (int k=ghostWidth; k<ksize-ghostWidth; k++)
	for (int j=ghostWidth; j<jsize-ghostWidth; j++)
	  for (int i=ghostWidth; i<isize-ghostWidth; i++) {
	    
	    if (i < isize/2) {
	      h_U(i,j,ID) = 1.0f;
	      h_U(i,j,IP) = 1.0f/(_gParams.gamma0-1.0f);
	    } else {
	      h_U(i,j,ID) = 0.125f;
	      h_U(i,j,IP) = 0.1f/(_gParams.gamma0-1.0f);
	    }
	    h_U(i,j,IU)=0.0f;
	    h_U(i,j,IV)=0.0f;
	    h_U(i,j,IW)=0.0f;
	    h_U(i,j,IA)=0.0f;
	    h_U(i,j,IB)=0.0f;
	    h_U(i,j,IC)=0.0f;
	  }
      
    } // end THREE_D

  } // MHDRunBase::init_mhd_sod

  // =======================================================
  // =======================================================
  /**
   * BrioWu test.
   * see http://flash.uchicago.edu/website/codesupport/flash3_ug_3p2/node31.html
   */
  void MHDRunBase::init_mhd_BrioWu()
  {    

    if (!mhdEnabled) {
      std::cerr << "MHD must be enabled to use this initial conditions !!!";
      return;
    }

    /* initial condition in grid interior */
    memset(h_U.data(),0,h_U.sizeBytes());

    /* get magnetic field strength */
    real_t B0 = configMap.getFloat("BrioWu","B0",1.0);
    real_t B1 = configMap.getFloat("BrioWu","B1",0.75);

    /* get densities */
    real_t d0 = configMap.getFloat("BrioWu","d0",1.0);
    real_t d1 = configMap.getFloat("BrioWu","d1",0.125);

    /* pressure */
    real_t p0 = 1.0;
    real_t p1 = 0.1;

    /* get direction : 0->X, 1->Y, 2->Z, 3->XY */
    int direction = configMap.getInteger("BrioWu","direction",0);
    if (direction < 0 || direction > 4)
      direction = 0;

    if (dimType == TWO_D) {

      /* init */
      if (direction == 0) { // along X

	for (int j=ghostWidth; j<jsize-ghostWidth; j++)
	  for (int i=ghostWidth; i<isize-ghostWidth; i++) {
	    
	    if (i < isize/2) {
	      h_U(i,j,ID) = d0;
	      h_U(i,j,IP) = p0/(_gParams.gamma0-1.0f) + 0.5 * (B0*B0 + B1*B1);
	      h_U(i,j,IA) = B1;
	      h_U(i,j,IB) = B0;
	    } else {
	      h_U(i,j,ID) = d1;
	      h_U(i,j,IP) = p1/(_gParams.gamma0-1.0f) + 0.5 * (B0*B0 + B1*B1);
	      h_U(i,j,IA) = B1;
	      h_U(i,j,IB) =-B0;
	    }
	    h_U(i,j,IU)=0.0f;
	    h_U(i,j,IV)=0.0f;
	    h_U(i,j,IW)=0.0f;
	    h_U(i,j,IC)=0.0f;
	  } // end for i
	
      } else if (direction == 1) { // along Y

	for (int j=ghostWidth; j<jsize-ghostWidth; j++)
	  for (int i=ghostWidth; i<isize-ghostWidth; i++) {
	    
	    if (j < jsize/2) {
	      h_U(i,j,ID) = d0;
	      h_U(i,j,IP) = p0/(_gParams.gamma0-1.0f) + 0.5 * (B0*B0 + B1*B1);
	      h_U(i,j,IA) = B0;
	      h_U(i,j,IB) = B1;
	    } else {
	      h_U(i,j,ID) = d1;
	      h_U(i,j,IP) = p1/(_gParams.gamma0-1.0f) + 0.5 * (B0*B0 + B1*B1);
	      h_U(i,j,IA) =-B0;
	      h_U(i,j,IB) = B1;
	    }
	    h_U(i,j,IU)=0.0f;
	    h_U(i,j,IV)=0.0f;
	    h_U(i,j,IW)=0.0f;
	    h_U(i,j,IC)=0.0f;
	  } // end for i

      } else if (direction == 3) { // along diagonal XY

	for (int j=ghostWidth; j<jsize-ghostWidth; j++)
	  for (int i=ghostWidth; i<isize-ghostWidth; i++) {
	    
	    if (1.0*i/isize+1.0*j/jsize < 1) {
	      h_U(i,j,ID) = d0;
	      h_U(i,j,IP) = p0/(_gParams.gamma0-1.0f) + 0.5 * ((-B0+B1)*(-B0+B1)/2 + (B0+B1)*(B0+B1)/2);
	      h_U(i,j,IA) =-B0/SQRT(2.) + B1/SQRT(2.);
	      h_U(i,j,IB) = B0/SQRT(2.) + B1/SQRT(2.);
	    } else {
	      h_U(i,j,ID) = d1;
	      h_U(i,j,IP) = p1/(_gParams.gamma0-1.0f) + 0.5 * ((-B0+B1)*(-B0+B1)/2 + (B0+B1)*(B0+B1)/2);
	      h_U(i,j,IA) = B0/SQRT(2.) + B1/SQRT(2.);
	      h_U(i,j,IB) =-B0/SQRT(2.) + B1/SQRT(2.);
	    }
	    h_U(i,j,IU)=0.0f;
	    h_U(i,j,IV)=0.0f;
	    h_U(i,j,IW)=0.0f;
	    h_U(i,j,IC)=0.0f;
	  } // end for i

      } else if (direction == 4) { // in lower-left corner (circle shape)

	for (int j=ghostWidth; j<jsize-ghostWidth; j++)
	  for (int i=ghostWidth; i<isize-ghostWidth; i++) {
	    
	    real_t phi = atan2(1.0*j,1.0*i);

	    if ( 1.0*i*i/(isize*isize) + 1.0*j*j/(jsize*jsize) < 1.0/4) {
	      h_U(i,j,ID) =  d0;
	      h_U(i,j,IA) = -B0*sin(phi) + B1*cos(phi);
	      h_U(i,j,IB) =  B0*cos(phi) + B1*sin(phi);
	      h_U(i,j,IP) =  p0/(_gParams.gamma0-1.0f) + 0.5 * (h_U(i,j,IA)*h_U(i,j,IA) + h_U(i,j,IB)*h_U(i,j,IB));
	    } else {
	      h_U(i,j,ID) =  d1;
	      h_U(i,j,IA) =  B0*sin(phi) + B1*cos(phi);
	      h_U(i,j,IB) = -B0*cos(phi) + B1*sin(phi);
	      h_U(i,j,IP) =  p1/(_gParams.gamma0-1.0f) + 0.5 * (h_U(i,j,IA)*h_U(i,j,IA) + h_U(i,j,IB)*h_U(i,j,IB));
	    }
	    h_U(i,j,IU)=0.0f;
	    h_U(i,j,IV)=0.0f;
	    h_U(i,j,IW)=0.0f;
	    h_U(i,j,IC)=0.0f;
	  } // end for i

      }

    } else if (dimType == THREE_D) { // THREE_D
      
      if (direction == 0) { // along X
	
	for (int k=ghostWidth; k<ksize-ghostWidth; k++)
	  for (int j=ghostWidth; j<jsize-ghostWidth; j++)
	    for (int i=ghostWidth; i<isize-ghostWidth; i++) {
	      
	      if (i < isize/2) {
		h_U(i,j,k,ID) = d0;
		h_U(i,j,k,IP) = p0/(_gParams.gamma0-1.0f) + 0.5 * (B0*B0 + B0*B0 + B1*B1);
		h_U(i,j,k,IA) = B1;
		h_U(i,j,k,IB) = B0;
		h_U(i,j,k,IC) = B0;
	      } else {
		h_U(i,j,k,ID) = d1;
		h_U(i,j,k,IP) = p1/(_gParams.gamma0-1.0f) + 0.5 * (B0*B0 + B0*B0 + B1*B1);
		h_U(i,j,k,IA) = B1;
		h_U(i,j,k,IB) =-B0;
		h_U(i,j,k,IC) =-B0;
	      }
	      h_U(i,j,k,IU)=0.0f;
	      h_U(i,j,k,IV)=0.0f;
	      h_U(i,j,k,IW)=0.0f;
	    } // end for i
	
      } else if (direction == 1) { // along Y

	for (int k=ghostWidth; k<ksize-ghostWidth; k++)
	  for (int j=ghostWidth; j<jsize-ghostWidth; j++)
	    for (int i=ghostWidth; i<isize-ghostWidth; i++) {
	      
	      if (j < jsize/2) {
		h_U(i,j,k,ID) = d0;
		h_U(i,j,k,IP) = p0/(_gParams.gamma0-1.0f) + 0.5 * (B0*B0 + B0*B0 + B1*B1);
		h_U(i,j,k,IA) = B0;
		h_U(i,j,k,IB) = B1;
		h_U(i,j,k,IC) = B0;
	      } else {
		h_U(i,j,k,ID) = d1;
		h_U(i,j,k,IP) = p1/(_gParams.gamma0-1.0f) + 0.5 * (B0*B0 + B0*B0 + B1*B1);
		h_U(i,j,k,IA) = -B0;
		h_U(i,j,k,IB) =  B1;
		h_U(i,j,k,IC) = -B0;
	      }
	      h_U(i,j,k,IU) = 0.0f;
	      h_U(i,j,k,IV) = 0.0f;
	      h_U(i,j,k,IW) = 0.0f;
	    } // end for i

      } else if (direction == 2) { // along Z

	for (int k=ghostWidth; k<ksize-ghostWidth; k++)
	  for (int j=ghostWidth; j<jsize-ghostWidth; j++)
	    for (int i=ghostWidth; i<isize-ghostWidth; i++) {
	      
	      if (k < ksize/2) {
		h_U(i,j,k,ID) = d0;
		h_U(i,j,k,IP) = p0/(_gParams.gamma0-1.0f) + 0.5 * (B0*B0 + B0*B0 + B1*B1);
		h_U(i,j,k,IA) = B0;
		h_U(i,j,k,IB) = B0;
		h_U(i,j,k,IC) = B1;
	      } else {
		h_U(i,j,k,ID) = d1;
		h_U(i,j,k,IP) = p1/(_gParams.gamma0-1.0f) + 0.5 * (B0*B0 + B0*B0 + B1*B1);
		h_U(i,j,k,IA) = -B0;
		h_U(i,j,k,IB) = -B0;
		h_U(i,j,k,IC) =  B1;
	      }
	      h_U(i,j,k,IU) = 0.0f;
	      h_U(i,j,k,IV) = 0.0f;
	      h_U(i,j,k,IW) = 0.0f;
	    } // end for i

      } else if (direction == 3) { // along XYZ

	for (int k=ghostWidth; k<ksize-ghostWidth; k++)
	  for (int j=ghostWidth; j<jsize-ghostWidth; j++)
	    for (int i=ghostWidth; i<isize-ghostWidth; i++) {

	      if (1.0*i/isize+1.0*j/jsize+1.0*k/ksize < 1) {
		h_U(i,j,k,ID) = d0;
		h_U(i,j,k,IA) = B1/SQRT(3.);
		h_U(i,j,k,IB) = B1/SQRT(3.)+B0*SQRT(2.0/3);
		h_U(i,j,k,IC) = B1/SQRT(3.)-2*B0/SQRT(6.0);
		h_U(i,j,k,IP) = p0/(_gParams.gamma0-1.0f) + 
		  0.5 * (h_U(i,j,k,IA) * h_U(i,j,k,IA) +
			 h_U(i,j,k,IB) * h_U(i,j,k,IB) +
			 h_U(i,j,k,IC) * h_U(i,j,k,IC) );
	      } else {
		h_U(i,j,k,ID) = d1;
		h_U(i,j,k,IA) = B1/SQRT(3.);
		h_U(i,j,k,IB) = B1/SQRT(3.)-B0*SQRT(2.0/3);
		h_U(i,j,k,IC) = B1/SQRT(3.)+2*B0/SQRT(6.0);
		h_U(i,j,k,IP) = p1/(_gParams.gamma0-1.0f) + 
		  0.5 * (h_U(i,j,k,IA) * h_U(i,j,k,IA) +
			 h_U(i,j,k,IB) * h_U(i,j,k,IB) +
			 h_U(i,j,k,IC) * h_U(i,j,k,IC) );
	      }
	      h_U(i,j,k,IU) = 0.0f;
	      h_U(i,j,k,IV) = 0.0f;
	      h_U(i,j,k,IW) = 0.0f;
	    } // end for i

      } // end direction
      
    } // end THREE_D

  } // MHDRunBase::init_mhd_BrioWu

  // =======================================================
  // =======================================================
  /**
   * The two-dimensional MHD rotor problem (Balsara and Spicer, 
   * 1999, JCP, 149, 270).
   * G. Toth, "The div(B)=0 constraint in shock-capturing MHD codes",
   * JCP, 161, 605 (2000)
   * 
   * Initial conditions are taken from Toth's paper.
   *
   * Initial conditions mentioned in the Flash user's guide are eroneous !
   * http://flash.uchicago.edu/website/codesupport/flash3_ug_3p3/node32.html#SECTION08123000000000000000
   *
   */
  void MHDRunBase::init_mhd_rotor()
  {

    if (!mhdEnabled) {
      std::cerr << "MHD must be enabled to use this initial conditions !!!";
      return;
    }

    /* first set zero everywhere */
    memset(h_U.data(),0,h_U.sizeBytes());

    // some constants
    const real_t FourPi = (real_t) (8.0*asin(1.0));
    real_t r0 = configMap.getFloat("rotor","r0",0.1f);
    real_t r1 = configMap.getFloat("rotor","r1",0.115f);
    real_t u0 = configMap.getFloat("rotor","u0",2.0f);
    real_t p0 = configMap.getFloat("rotor","p0",1.0f);
    real_t b0 = configMap.getFloat("rotor","b0",5.0/SQRT(FourPi));

    real_t &xMin = _gParams.xMin;
    real_t &yMin = _gParams.yMin;
    real_t  xMax = configMap.getFloat("mesh","xmax",1.0);
    real_t  yMax = configMap.getFloat("mesh","ymax",1.0);

    real_t xCenter = (xMax + xMin)/2;
    real_t yCenter = (yMax + yMin)/2;

    real_t &gamma = _gParams.gamma0;
 
    if (dimType == TWO_D) {

      for (int j=ghostWidth; j<jsize-ghostWidth; j++) {
	
	real_t yPos = yMin + dx/2 + (j-ghostWidth)*dy;
	
	for (int i=ghostWidth; i<isize-ghostWidth; i++) {

	  real_t xPos = xMin + dx/2 + (i-ghostWidth)*dx;
	  real_t r = SQRT( (xPos-xCenter)*(xPos-xCenter) + (yPos-yCenter)*(yPos-yCenter) );
	  real_t f_r = (r1-r)/(r1-r0);

	  if (r<=r0) {
	    h_U(i,j,ID) = 10.0;
	    h_U(i,j,IU) = -u0*(yPos-yCenter)/r0;
	    h_U(i,j,IV) =  u0*(xPos-xCenter)/r0;
	  } else if (r<=r1) {
	    h_U(i,j,ID) = 1+9*f_r;
	    h_U(i,j,IU) = -f_r*u0*(yPos-yCenter)/r;
	    h_U(i,j,IV) =  f_r*u0*(xPos-xCenter)/r;
	  } else {
	    h_U(i,j,ID) = 1.0;
	    h_U(i,j,IU) = 0.0;
	    h_U(i,j,IV) = 0.0;
	  }
	  h_U(i,j,IW) = 0.0;
	  h_U(i,j,IA) = b0; //5.0/SQRT(FourPi);
	  h_U(i,j,IB) = 0.0;
	  h_U(i,j,IC) = 0.0;
	  h_U(i,j,IP) = p0/(gamma-1.0) + 
	    ( h_U(i,j,IU)*h_U(i,j,IU) + 
	      h_U(i,j,IV)*h_U(i,j,IV) +
	      h_U(i,j,IW)*h_U(i,j,IW) )/2/h_U(i,j,ID) +
	    ( h_U(i,j,IA)*h_U(i,j,IA))/2;

	} // end for i

      } // end for j

    } else if (dimType == THREE_D) {

    } // end THREE_D

  } // MHDRunBase::init_mhd_rotor

  // =======================================================
  // =======================================================
  /**
   * The 2D/3D MHD field loop advection problem.
   * 
   * Parameters that can be set in the ini file :
   * - radius       : radius of field loop
   * - amplitude    : amplitude of vector potential (and therefore B in loop)
   * - vflow        : flow velocity
   * - densityRatio : density ratio in loop.  Enables density advection and
   *                  thermal conduction tests.
   * The flow is automatically set to run along the diagonal. 
   * - direction : integer 
   *   direction 0 -> field loop in x-y plane (cylinder in 3D)
   *   direction 1 -> field loop in y-z plane (cylinder in 3D)
   *   direction 2 -> field loop in z-x plane (cylinder in 3D)
   *   direction 3 -> rotated cylindrical field loop in 3D.
   *
   * Reference :
   * - T. Gardiner & J.M. Stone, "An unsplit Godunov method for ideal MHD
   *   via constrined transport", JCP, 205, 509 (2005)
   * - http://www.astro.princeton.edu/~jstone/Athena/tests/field-loop/Field-loop.html
   */
  void MHDRunBase::init_mhd_field_loop()
  {

    if (!mhdEnabled) {
      std::cerr << "MHD must be enabled to use this initial conditions !!!";
      return;
    }

    /* first set zero everywhere */
    memset(h_U.data(),0,h_U.sizeBytes());

    // some constants
    real_t radius    = configMap.getFloat("FieldLoop","radius"   ,1.0f);
    real_t density_in= configMap.getFloat("FieldLoop","density_in", 1.0f);
    real_t amplitude = configMap.getFloat("FieldLoop","amplitude",1.0f);
    real_t vflow     = configMap.getFloat("FieldLoop","vflow"    ,1.0f);

    real_t &xMin = _gParams.xMin;
    real_t &yMin = _gParams.yMin;
    //real_t &zMin = _gParams.zMin;

    const real_t cos_theta = 2.0/sqrt(5.0);
    const real_t sin_theta = sqrt(1-cos_theta*cos_theta);
    
    if (dimType == TWO_D) {

      // vector potential
      HostArray<real_t> Az;
      Az.allocate( make_uint3(isize, jsize, 1) );

      // initialize vector potential
      for (int j=ghostWidth; j<jsize-ghostWidth+1; j++) {
	
	real_t yPos = yMin + dy/2 + (j-ghostWidth)*dy;
	
	for (int i=ghostWidth; i<isize-ghostWidth+1; i++) {
	  
	  real_t xPos = xMin + dx/2 + (i-ghostWidth)*dx;

	  real_t r = SQRT(xPos*xPos+yPos*yPos);
	  if ( r < radius ) {
	    Az(i,j,0) = amplitude * ( radius - r );
	  } else {
	    Az(i,j,0) = ZERO_F;
	  }
	
	} // end for i

      } // end for j

      // init MHD
      for (int j=ghostWidth; j<jsize-ghostWidth; j++) {
	real_t yPos = yMin + dy/2 + (j-ghostWidth)*dy;
	for (int i=ghostWidth; i<isize-ghostWidth; i++) {
	  real_t xPos = xMin + dx/2 + (i-ghostWidth)*dx;

	  real_t diag = SQRT(1.0*(nx*nx + ny*ny + nz*nz));
	  real_t r    = SQRT(xPos*xPos+yPos*yPos);

	  // density
	  if (r < radius)
	    h_U(i,j,ID) = density_in;
	  else
	    h_U(i,j,ID) = 1.0f;

	  // rho*vx
	  //h_U(i,j,IU) = h_U(i,j,ID)*vflow*nx/diag;
	  h_U(i,j,IU) = h_U(i,j,ID)*vflow*cos_theta;

	  // rho*vy
	  //h_U(i,j,IV) = h_U(i,j,ID)*vflow*ny/diag;
	  h_U(i,j,IV) = h_U(i,j,ID)*vflow*sin_theta;

	  // rho*vz
	  h_U(i,j,IW) = h_U(i,j,ID)*vflow*nz/diag; //ZERO_F;

	  // bx
	  h_U(i,j,IA) =   (Az(i  ,j+1,0) - Az(i,j,0))/dy;

	  // by
	  h_U(i,j,IB) = - (Az(i+1,j  ,0) - Az(i,j,0))/dx;

	  // bz
	  h_U(i,j,IC) = ZERO_F;

	  // total energy
	  h_U(i,j,IP) = 1.0f/(_gParams.gamma0-1.0f) + 
	    0.5 * (h_U(i,j,IA) * h_U(i,j,IA) + h_U(i,j,IB) * h_U(i,j,IB)) +
	    0.5 * (h_U(i,j,IU) * h_U(i,j,IU) + h_U(i,j,IV) * h_U(i,j,IV))/h_U(i,j,ID);
	  //h_U(i,j,IP) = ZERO_F;
	} // end for i
      } // end for j

    } else if (dimType == THREE_D) {

      // vector potential
      HostArray<real_t> A;
      A.allocate( make_uint4(isize, jsize, ksize, 3) );

      // amplitude of a random noise added to the vector potential
      const double amp       = configMap.getFloat("FieldLoop","amp",0.01);  
      const int    seed      = configMap.getInteger("FieldLoop","seed",0);
      srand48(seed);

      for (int k=0; k<ksize; k++) {
	
	//real_t zPos = zMin + dz/2 + (k-ghostWidth)*dz;
	
	for (int j=0; j<jsize; j++) {
	  
	  real_t yPos = yMin + dy/2 + (j-ghostWidth)*dy;
	  
	  for (int i=0; i<isize; i++) {
	    
	    real_t xPos = xMin + dx/2 + (i-ghostWidth)*dx;
	    
	    A(i,j,k,0) = ZERO_F;
	    A(i,j,k,1) = ZERO_F;
	    A(i,j,k,2) = ZERO_F + amp*(drand48()-0.5);
	    real_t r    = SQRT(xPos*xPos+yPos*yPos);
	    if (r < radius) {
	      A(i,j,k,2) = amplitude * (radius - r);
	    }

	  } // end for i

	} // end for j
	
      } // end for k
      
      // init MHD
      for (int k=ghostWidth; k<ksize-ghostWidth; k++) {
	//real_t zPos = zMin + dz/2 + (k-ghostWidth)*dz;

	for (int j=ghostWidth; j<jsize-ghostWidth; j++) {
	  real_t yPos = yMin + dy/2 + (j-ghostWidth)*dy;

	  for (int i=ghostWidth; i<isize-ghostWidth; i++) {
	    real_t xPos = xMin + dx/2 + (i-ghostWidth)*dx;

	    //real_t diag = SQRT(1.0*(nx*nx + ny*ny + nz*nz));
	    real_t r;
	    r = SQRT(xPos*xPos + yPos*yPos);
	    
	    // density
	    if (r < radius)
	      h_U(i,j,k,ID) = density_in;
	    else
	      h_U(i,j,k,ID) = 1.0f;
	    
	    // rho*vx
	    //h_U(i,j,k,IU) = h_U(i,j,k,ID)*vflow*nx/diag;
	    h_U(i,j,k,IU) = h_U(i,j,k,ID)*vflow*cos_theta;
	    
	    // rho*vy
	    //h_U(i,j,k,IV) = h_U(i,j,k,ID)*vflow*ny/diag;
	    h_U(i,j,k,IV) = h_U(i,j,k,ID)*vflow*sin_theta;
	    
	    // rho*vz
	    h_U(i,j,k,IW) = ZERO_F; //h_U(i,j,k,ID)*vflow*nz/diag;
	    
	    // bx
	    h_U(i,j,k,IA) =
	      ( A(i,j+1,k  ,2) - A(i,j,k,2) ) / dy -
	      ( A(i,j  ,k+1,1) - A(i,j,k,1) ) / dz;
	    
	    // by
	    h_U(i,j,k,IB) = 
	      ( A(i  ,j,k+1,0) - A(i,j,k,0) ) / dz -
	      ( A(i+1,j,k  ,2) - A(i,j,k,2) ) / dx;
	    
	    // bz
	    h_U(i,j,k,IC) = 
	      ( A(i+1,j  ,k,1) - A(i,j,k,1) ) / dx -
	      ( A(i  ,j+1,k,0) - A(i,j,k,0) ) / dy;
	    
	    // total energy
	    if (_gParams.cIso>0) {
	      h_U(i,j,k,IP) = ZERO_F;
	    } else {
	      h_U(i,j,k,IP) = 1.0f/(_gParams.gamma0-1.0f) + 
		0.5 * (h_U(i,j,k,IA) * h_U(i,j,k,IA)  + 
		       h_U(i,j,k,IB) * h_U(i,j,k,IB)  +
		       h_U(i,j,k,IC) * h_U(i,j,k,IC)) +
		0.5 * (h_U(i,j,k,IU) * h_U(i,j,k,IU) + 
		       h_U(i,j,k,IV) * h_U(i,j,k,IV) +
		       h_U(i,j,k,IW) * h_U(i,j,k,IW))/h_U(i,j,k,ID);
	    }
	  } // end for i
	} // end for j
      } // end for k
      
    } // end THREE_D
    
  } // MHDRunBase::init_mhd_field_loop

  // =======================================================
  // =======================================================
  /**
   * The 2D/3D MHD current sheet problem.
   * 
   * Parameters that can be set in the ini file :
   * - A      : amplitude of velocity wave
   * - B0     : amplitude of the Y component of magnetic field
   * - beta   : total energy
   *
   * Reference :
   * - http://www.astro.princeton.edu/~jstone/Athena/tests/current-sheet/current-sheet.html
   * \sa Dumses in patch/tests/sheet
   */
  void MHDRunBase::init_mhd_current_sheet()
  {

    if (!mhdEnabled) {
      std::cerr << "MHD must be enabled to use this initial conditions !!!";
      return;
    }

    /* first set zero everywhere */
    memset(h_U.data(),0,h_U.sizeBytes());

    // some constants
    real_t A    = configMap.getFloat("CurrentSheet","A" ,0.1f);
    real_t B0   = configMap.getFloat("CurrentSheet","B0",1.0f);
    real_t beta = configMap.getFloat("CurrentSheet","beta",0.1f);

    real_t &xMin = _gParams.xMin;
    real_t &yMin = _gParams.yMin;
    //real_t &zMin = _gParams.zMin;
    
    if (dimType == TWO_D) {

      for (int j=0; j<jsize; j++) {
	for (int i=0; i<isize; i++) {

	  real_t xPos = xMin + dx/2 + (i-ghostWidth)*dx;
	  real_t yPos = yMin + dy/2 + (j-ghostWidth)*dy;

	  h_U(i,j,ID)=ONE_F;
	  h_U(i,j,IP)=beta;
	  h_U(i,j,IU)=h_U(i,j,ID)*A*sin(M_PI*yPos);
	  h_U(i,j,IV)=ZERO_F;
	  h_U(i,j,IW)=ZERO_F;
	  h_U(i,j,IA)=ZERO_F;
	  h_U(i,j,IB)= (xPos<0.5 or xPos>1.5) ? B0 : -B0;
	  h_U(i,j,IC)=ZERO_F;
	} // end for i
      } // end for j

    } else { // THREE_D

      for (int k=0; k<ksize; k++) {
	for (int j=0; j<jsize; j++) {
	  for (int i=0; i<isize; i++) {
	    
	    real_t xPos = xMin + dx/2 + (i-ghostWidth)*dx;
	    real_t yPos = yMin + dy/2 + (j-ghostWidth)*dy;
	    
	    h_U(i,j,k,ID)=ONE_F;
	    h_U(i,j,k,IP)=beta;
	    h_U(i,j,k,IU)=h_U(i,j,ID)*A*sin(M_PI*yPos);
	    h_U(i,j,k,IV)=ZERO_F;
	    h_U(i,j,k,IW)=ZERO_F;
	    h_U(i,j,k,IA)=ZERO_F;
	    h_U(i,j,k,IB)= (xPos<0.5 or xPos>1.5) ? B0 : -B0;
	    h_U(i,j,k,IC)=ZERO_F;

	  } // end for i
	} // end for j     
      } // end for k

    } // end THREE_D

  } // MHDRunBase::init_mhd_current_sheet

  // =======================================================
  // =======================================================
  /**
   * The 2D/3D MHD inertial wave problem (test for Omega0 dependent terms).
   * 
   * This test aims at verifying that the numerical scheme is OK when we add rotating
   * frame terms.
   *
   * Parameters that can be set in the ini file :
   * - density
   * - delta_vx : amplitude of the X component of velocity field
   *
   * \sa Dumses in patch/tests/inertial_wave
   */
  void MHDRunBase::init_mhd_inertial_wave()
  {

    if (!mhdEnabled) {
      std::cerr << "MHD must be enabled to use this initial conditions !!!";
      return;
    }

    /* first set zero everywhere */
    memset(h_U.data(),0,h_U.sizeBytes());

    // some constants
    real_t density  = configMap.getFloat("InertialWave","density" ,1.0f);
    real_t energy   = configMap.getFloat("InertialWave","energy",1.0f);
    real_t delta_vx = configMap.getFloat("InertialWave","delta_vx",1.0f);
    delta_vx *= _gParams.cIso;

    if (dimType == TWO_D) {

      for (int j=0; j<jsize; j++) {
	for (int i=0; i<isize; i++) {

	  h_U(i,j,ID)=density;
	  h_U(i,j,IP)=energy;
	  h_U(i,j,IU)=density*delta_vx;
	  h_U(i,j,IV)=ZERO_F;
	  h_U(i,j,IW)=ZERO_F;
	  h_U(i,j,IA)=ZERO_F;
	  h_U(i,j,IB)=ZERO_F;
	  h_U(i,j,IC)=ZERO_F;

	} // end for i
      } // end for j

    } else { // THREE_D

      for (int k=0; k<ksize; k++) {
	for (int j=0; j<jsize; j++) {
	  for (int i=0; i<isize; i++) {
	    
	    h_U(i,j,k,ID)=density;
	    h_U(i,j,k,IP)=energy;
	    h_U(i,j,k,IU)=density*delta_vx;
	    h_U(i,j,k,IV)=ZERO_F;
	    h_U(i,j,k,IW)=ZERO_F;
	    h_U(i,j,k,IA)=ZERO_F;
	    h_U(i,j,k,IB)=ZERO_F;
	    h_U(i,j,k,IC)=ZERO_F;

	  } // end for i
	} // end for j     
      } // end for k

    } // end THREE_D

  } // MHDRunBase::init_mhd_inertial_wave

  // =======================================================
  // =======================================================
  /**
   * The 2D/3D MHD shear wave problem (Shearing border conditions must be activated).
   * 
   * This test aims at verifying that the numerical scheme is OK with shearing box 
   * border conditions.
   *
   * Parameters that can be set in the ini file :
   * - density
   * - delta_vx : amplitude of the X component of velocity field
   *
   * \sa Dumses in patch/tests/shwave
   */
  void MHDRunBase::init_mhd_shear_wave()
  {

    if (!mhdEnabled) {
      std::cerr << "MHD must be enabled to use this initial conditions !!!";
      return;
    }

    if (boundary_xmin != BC_SHEARINGBOX or boundary_xmax != BC_SHEARINGBOX) {
      std::cerr << "Shearing box border conditions must enabled along X-direction !!!\n";
      return;
    }

    /* first set zero everywhere */
    memset(h_U.data(),0,h_U.sizeBytes());

    // some constants
    const double TwoPi     = 4.0*asin(1.0);
    const double d0        = 1.0;

    const double Lx        = _gParams.dx * _gParams.nx;
    const double Ly        = _gParams.dy * _gParams.ny;

    //const double density   = configMap.getFloat("ShearWave","density" ,1.0);
    const double energy    = configMap.getFloat("ShearWave","energy",1.0);
    const double delta_vx  = (-4.0e-4) *_gParams.cIso;
    const double delta_vy  = ( 1.0e-4) *_gParams.cIso;
    const double kx0       = -4*TwoPi/Lx;
    const double ky0       =    TwoPi/Ly;
    const double xi0       = 0.5* _gParams.Omega0/d0;
    const double delta_rho = (kx0*delta_vy-ky0*delta_vx)/xi0;

    real_t &xMin = _gParams.xMin;
    real_t &yMin = _gParams.yMin;

    if (dimType == TWO_D) {

      for (int j=0; j<jsize; j++) {
	
	double yPos = yMin + dy/2 + (j-ghostWidth)*dy;

	for (int i=0; i<isize; i++) {

	  double xPos = xMin + dx/2 + (i-ghostWidth)*dx;

	  h_U(i,j,ID) = d0*(1.0-delta_rho*sin(kx0*xPos+ky0*yPos));
	  h_U(i,j,IP) = energy;
	  h_U(i,j,IU) = h_U(i,j,ID)*delta_vx*cos(kx0*xPos+ky0*yPos);
	  h_U(i,j,IV) = h_U(i,j,ID)*delta_vy*cos(kx0*xPos+ky0*yPos);
	  h_U(i,j,IW) = ZERO_F;
	  h_U(i,j,IA) = ZERO_F;
	  h_U(i,j,IB) = ZERO_F;
	  h_U(i,j,IC) = ZERO_F;

	} // end for i
      } // end for j

    } else { // THREE_D

      for (int k=0; k<ksize; k++) {
	
	for (int j=0; j<jsize; j++) {

	  double yPos = yMin + dy/2 + (j-ghostWidth)*dy;
	  
	  for (int i=0; i<isize; i++) {
	    
	    double xPos = xMin + dx/2 + (i-ghostWidth)*dx;
	    
	    h_U(i,j,k,ID) = d0*(1.0-delta_rho*sin(kx0*xPos+ky0*yPos));
	    h_U(i,j,k,IP) = energy;
	    h_U(i,j,k,IU) = h_U(i,j,ID)*delta_vx*cos(kx0*xPos+ky0*yPos);
	    h_U(i,j,k,IV) = h_U(i,j,ID)*delta_vy*cos(kx0*xPos+ky0*yPos);
	    h_U(i,j,k,IW) = ZERO_F;
	    h_U(i,j,k,IA) = ZERO_F;
	    h_U(i,j,k,IB) = ZERO_F;
	    h_U(i,j,k,IC) = ZERO_F;
	    
	  } // end for i
	} // end for j     
      } // end for k

    } // end THREE_D

  } // MHDRunBase::init_mhd_shear_wave

  // =======================================================
  // =======================================================
  /**
   * The 3D MHD MRI problem (Shearing border conditions must be activated).
   * 
   * Setup for making a MRI (Magneto-Rotational Instability) simulation.
   *
   * Parameters that can be set in the ini file :
   * - density
   * - beta : real - large value means small magnetic field  /
   *                 small magnetic energy compared to thermal energy
   * - type : string ('noflux', 'pyl', 'fluxZ')
   * - amp  : real - velocity amplitude
   * - seed : int - random number seed
   *
   * \sa Dumses in patch/mri
   */
  void MHDRunBase::init_mhd_mri()
  {

    if (!mhdEnabled) {
      std::cerr << "MHD must be enabled to use this initial conditions !!!";
      return;
    }

    if (dimType == TWO_D) {
      std::cerr << "MRI simulations is only available in 3D !\n";
      return;
    }
    
    if (boundary_xmin != BC_SHEARINGBOX or boundary_xmax != BC_SHEARINGBOX) {
      std::cerr << "Shearing box border conditions must enabled along X-direction !!!\n";
      return;
    }

    // Check if Isothermal EOS is enabled, if not print a warning
    if (_gParams.cIso <= 0) {
      std::cout << "############################################################\n";
      std::cout << "#### WARNING: MRI problem called with NON Isothermal EOS !!!\n";
      std::cout << "############################################################\n";
    }


    /* first set zero everywhere */
    memset(h_U.data(),0,h_U.sizeBytes());

    // some constants
    const double TwoPi     = 4.0*asin(1.0);
    const double d0        = configMap.getFloat("MRI","density",1.0);
    const double beta      = configMap.getFloat("MRI", "beta", 400.0);

    const double p0        = d0 * _gParams.cIso * _gParams.cIso; 
    
    real_t &zMin = _gParams.zMin;
    real_t  zMax = configMap.getFloat("mesh","zmax",1.0);
    real_t &Omega0 = _gParams.Omega0;

    double B0;
    std::string type       = configMap.getString("MRI","type","noflux"); 
    if (!type.compare("pyl"))
      B0 = 3.0/2.0 * sqrt( d0 * Omega0 * Omega0 * (zMax-zMin)*(zMax-zMin) / beta);
    else
      B0 = 2.0 * sqrt(p0/beta);

    const double amp       = configMap.getFloat("MRI","amp",0.01);  
    const int    seed      = configMap.getInteger("MRI","seed",0);
    const double d_amp     = configMap.getFloat("MRI","density_fluctuations",0.0);

    real_t &xMin = _gParams.xMin;

    // initialize random number generator
    srand48(seed);

    for (int k=0; k<ksize; k++) {
      
      for (int j=0; j<jsize; j++) {
	
	for (int i=0; i<isize; i++) {

	  double xPos = xMin + dx/2 + (i-ghostWidth)*dx;

	  h_U(i,j,k,ID) = d0 * (1 + d_amp*2*(drand48()-0.5));
	  h_U(i,j,k,IP) = 0;
	  h_U(i,j,k,IU) = d0*amp*(drand48()-0.5)*sqrt(p0);
	  h_U(i,j,k,IV) = d0*amp*(drand48()-0.5)*sqrt(p0);
	  h_U(i,j,k,IW) = d0*amp*(drand48()-0.5)*sqrt(p0);
	  h_U(i,j,k,IA) = ZERO_F;
	  h_U(i,j,k,IB) = ZERO_F; // B0 (Toroidal field)
	  if (!type.compare("noflux")) {
	    h_U(i,j,k,IC) = B0*sin(TwoPi*xPos);
	  } else if (!type.compare("pyl") || !type.compare("fluxZ")){
	    h_U(i,j,k,IC) = B0;
	  } else {
	    h_U(i,j,k,IC) = ZERO_F;
	  }
	  
	} // end for i
      } // end for j     
    } // end for k

    /*
     * if gravity is enabled, special init routine
     */
    if (gravityEnabled) {

      // initialize gravity field
      init_mhd_mri_grav_field();

      // modify density and magnetif field
      double zFloor      = configMap.getFloat("MRI", "zFloor"       , 5.0);
      real_t &cIso       = _gParams.cIso;
      double H           = cIso/Omega0;

      for (int k=0; k<ksize; k++) {
	real_t zPos = _gParams.zMin + dz/2 + (k-ghostWidth)*dz;
	
	for (int j=0; j<jsize; j++) {
	  //real_t yPos = _gParams.yMin + dy/2 + (j-ghostWidth)*dy;
	  
	  for (int i=0; i<isize; i++) {
	    //real_t xPos = _gParams.xMin + dx/2 + (i-ghostWidth)*dx;

	    // modify density
	    h_U(i,j,k,ID) = d0 * fmax ( exp(-(zPos*zPos)/2.0/(H*H)), exp(-zFloor*zFloor/2.0) );
	    //h_U(i,j,k,ID) = d0 * exp(-(zPos*zPos)/2.0/(H*H));

	    // enforce azimuthal magnetic field
	    h_U(i,j,k,IA) = ZERO_F;
	    h_U(i,j,k,IB) = ZERO_F;
	    h_U(i,j,k,IC) = ZERO_F;

	    if (zPos < H and zPos > -H)
	      h_U(i,j,k,IB) = B0;

	  } // end for i
	} // end for j
      } // end for k
   
    } // end gravity enabled
    
  } // MHDRunBase::init_mhd_mri

  // =======================================================
  // =======================================================
  /**
   * The 2D/3D MHD Kelvin-Helmholtz instability problem.
   *
   * Use the same init conditions as in Athena 4.1.
   *
   * \sa http://www.astro.princeton.edu/~jstone/Athena/tests/kh/kh.html
   * 
   * Domain extent : -0.5 0.5 in each direction
   *
   */
  void MHDRunBase::init_mhd_Kelvin_Helmholtz()
  {

    /* initial condition in grid interior */
    memset(h_U.data(),0,h_U.sizeBytes());

    /* initialize random generator */
    int seed = configMap.getInteger("kelvin-helmholtz", "seed", 1);
    srand(seed);

    /* initialize perturbation amplitude */
    real_t amplitude = configMap.getFloat("kelvin-helmholtz", "amplitude", 0.01);

    /* perturbation type sine / random */
    bool p_sine = configMap.getFloat("kelvin-helmholtz", "perturbation_sine", false);
    bool p_rand = configMap.getFloat("kelvin-helmholtz", "perturbation_rand", true);

    /* inner and outer fluid density */
    real_t rho_inner = configMap.getFloat("kelvin-helmholtz", "rho_inner", 2.0);
    real_t rho_outer = configMap.getFloat("kelvin-helmholtz", "rho_outer", 1.0);
    real_t pressure  = configMap.getFloat("kelvin_helmholtz", "pressure", 2.5);

    /* velocity amplitude */
    real_t v0        = configMap.getFloat("kelvin-helmholtz", "v0",        1.0);

    /* magnetic field */
    real_t b0        = configMap.getFloat("kelvin-helmholtz", "b0",        1.0);

    real_t &xMin = _gParams.xMin;
    real_t &yMin = _gParams.yMin;
    //real_t &zMin = _gParams.zMin;

    real_t &xMax = _gParams.xMax;
    real_t &yMax = _gParams.yMax;
    //real_t &zMax = _gParams.zMax;

    if (dimType == TWO_D) {

      for (int j=ghostWidth; j<jsize-ghostWidth; j++) {
	real_t yPos = yMin + (yMax-yMin)*j/jsize;
	//real_t yPos = yMin + dy/2 + (j-ghostWidth)*dy;

	for (int i=ghostWidth; i<isize-ghostWidth; i++) {
	  real_t xPos = xMin + (xMax-xMin)*j/jsize;
	  //real_t xPos = xMin + dx/2 + (i-ghostWidth)*dx;

	  if ( yPos < yMin+0.25*(yMax-yMin) or
	       yPos > yMin+0.75*(yMax-yMin) ) {
	    
	    h_U(i,j,ID) = rho_outer;

	    h_U(i,j,IU) = rho_outer *
	      (v0 + 
	       p_rand * amplitude * ( 1.0*rand()/RAND_MAX - 0.5 ) +
	       p_sine * amplitude * sin(2*M_PI*xPos) );
	    h_U(i,j,IV) = rho_outer *
	      (p_rand * amplitude * ( 1.0*rand()/RAND_MAX - 0.5 ) +
	       p_sine * amplitude * sin(2*M_PI*xPos) );
	    h_U(i,j,IW) = ZERO_F;

	    h_U(i,j,IA) = b0;
	    h_U(i,j,IB) = ZERO_F;
	    h_U(i,j,IC) = ZERO_F;
	    
	    /* Pressure scaled to give a sound speed of 1 with gamma=1.4 */
	    h_U(i,j,IP) = pressure/(_gParams.gamma0-1.0f) +
	      0.5 * (h_U(i,j,IU) * h_U(i,j,IU) + 
		     h_U(i,j,IV) * h_U(i,j,IV) +
		     h_U(i,j,IW) * h_U(i,j,IW))/h_U(i,j,ID) +
	      0.5 * b0 * b0;
	    
	  } else {
	  
	    h_U(i,j,ID) = rho_inner;

	    h_U(i,j,IU) = rho_inner * 
	      (-v0 + 
	       p_rand * amplitude * ( 1.0*rand()/RAND_MAX - 0.5 ) +
	       p_sine * amplitude * sin(2*M_PI*xPos) );
	    h_U(i,j,IV) =   rho_inner * 
	      (p_rand * amplitude * ( 1.0*rand()/RAND_MAX - 0.5 ) +
	       p_sine * amplitude * sin(2*M_PI*xPos) );
	    h_U(i,j,IW) =   ZERO_F;

	    h_U(i,j,IA) =   b0;
	    h_U(i,j,IB) =   ZERO_F;
	    h_U(i,j,IC) =   ZERO_F;
	    
	    /* Pressure scaled to give a sound speed of 1 with gamma=1.4 */
	    h_U(i,j,IP) = pressure/(_gParams.gamma0-1.0f) +
	      0.5 * (h_U(i,j,IU) * h_U(i,j,IU) + 
		     h_U(i,j,IV) * h_U(i,j,IV) +
		     h_U(i,j,IW) * h_U(i,j,IW))/h_U(i,j,ID) +
	      0.5 * b0 * b0;
	  }

	} // end for i
      } // end for j

    } else { // THREE_D

      for (int k=ghostWidth; k<ksize-ghostWidth; k++) {
	//real_t zPos = zMin + dz/2 + (k-ghostWidth)*dz;

	for (int j=ghostWidth; j<jsize-ghostWidth; j++) {
	  real_t yPos = yMin + dy/2 + (j-ghostWidth)*dy;
	  
	  for (int i=ghostWidth; i<isize-ghostWidth; i++) {
	    real_t xPos = xMin + dx/2 + (i-ghostWidth)*dx;
	    
	    if ( yPos < yMin+0.25*(yMax-yMin) or
		 yPos > yMin+0.75*(yMax-yMin) ) {
	      
	      h_U(i,j,k,ID) = rho_outer;
	      
	      h_U(i,j,k,IU) = rho_outer *
		(v0 + 
		 p_rand * amplitude * ( 1.0*rand()/RAND_MAX - 0.5 ) +
		 p_sine * amplitude * sin(2*M_PI*xPos) );
	      h_U(i,j,k,IV) = rho_outer *
		(p_rand * amplitude * ( 1.0*rand()/RAND_MAX - 0.5 ) +
		 p_sine * amplitude * sin(2*M_PI*xPos) );
	      h_U(i,j,k,IW) = rho_outer *
		(p_rand * amplitude * ( 1.0*rand()/RAND_MAX - 0.5 ) +
		 p_sine * amplitude * sin(2*M_PI*xPos) );
	      
	      h_U(i,j,k,IA) = b0;
	      h_U(i,j,k,IB) = ZERO_F;
	      h_U(i,j,k,IC) = ZERO_F;
	      
	      /* Pressure scaled to give a sound speed of 1 with gamma=1.4 */
	      h_U(i,j,k,IP) = pressure/(_gParams.gamma0-1.0f) +
		0.5 * (h_U(i,j,k,IU) * h_U(i,j,k,IU) + 
		       h_U(i,j,k,IV) * h_U(i,j,k,IV) +
		       h_U(i,j,k,IW) * h_U(i,j,k,IW))/h_U(i,j,k,ID) +
		0.5 * b0 * b0;
	      
	    } else {
	      
	      h_U(i,j,k,ID) = rho_inner;
	      
	      h_U(i,j,k,IU) = rho_inner * 
		(-v0 + 
		 p_rand * amplitude * ( 1.0*rand()/RAND_MAX - 0.5 ) +
		 p_sine * amplitude * sin(2*M_PI*xPos) );
	      h_U(i,j,k,IV) = rho_inner * 
		(p_rand * amplitude * ( 1.0*rand()/RAND_MAX - 0.5 ) +
		 p_sine * amplitude * sin(2*M_PI*xPos) );
	      h_U(i,j,k,IW) = rho_inner * 
		(p_rand * amplitude * ( 1.0*rand()/RAND_MAX - 0.5 ) +
		 p_sine * amplitude * sin(2*M_PI*xPos) );
	      
	      h_U(i,j,k,IA) =   b0;
	      h_U(i,j,k,IB) =   ZERO_F;
	      h_U(i,j,k,IC) =   ZERO_F;
	      
	      /* Pressure scaled to give a sound speed of 1 with gamma=1.4 */
	      h_U(i,j,k,IP) = pressure/(_gParams.gamma0-1.0f) +
		0.5 * (h_U(i,j,k,IU) * h_U(i,j,k,IU) + 
		       h_U(i,j,k,IV) * h_U(i,j,k,IV) +
		       h_U(i,j,k,IW) * h_U(i,j,k,IW))/h_U(i,j,k,ID) +
		0.5 * b0 * b0;
	    }
	    
	  } // end for i
	} // end for j
      } // end for k

    } // end THREE_D

  } // MHDRunBase::init_mhd_Kelvin_Helmholtz

  // =======================================================
  // =======================================================
  /**
   * The 2D/3D MHD Rayleigh-Taylor instability problem.
   *
   * See
   * http://www.astro.princeton.edu/~jstone/Athena/tests/rt/rt.html
   * for a description of such initial conditions
   */
  void MHDRunBase::init_mhd_Rayleigh_Taylor()
  {

    // magnetic field initial conditions
    real_t Bx0 = configMap.getFloat("rayleigh-taylor", "bx",  1e-8);
    real_t By0 = configMap.getFloat("rayleigh-taylor", "by",  1e-8);
    real_t Bz0 = configMap.getFloat("rayleigh-taylor", "bz",  1e-8);
    
    // call hydro initialization routine
    init_hydro_Rayleigh_Taylor();

    if (dimType == TWO_D) {
      
      // initialize magnetic field
      for (int j=0; j<jsize; j++)
	for (int i=0; i<isize; i++) {
	  h_U(i,j,IBX) = Bx0;
	  h_U(i,j,IBY) = By0;
	  h_U(i,j,IBZ) = Bz0;
	  
	  // update energy
	  h_U(i,j,IP) += 0.5 * (Bx0*Bx0 + By0*By0 + Bz0*Bz0);
	  
	} // end for i,j

    } else { //THREE_D
   
      // initialize magnetic field
      for (int k=0; k<ksize; k++)
	for (int j=0; j<jsize; j++)
	  for (int i=0; i<isize; i++) {
	    h_U(i,j,k,IBX) = Bx0;
	    h_U(i,j,k,IBY) = By0;
	    h_U(i,j,k,IBZ) = Bz0;
	    
	    // update energy
	    h_U(i,j,k,IP) += 0.5 * (Bx0*Bx0 + By0*By0 + Bz0*Bz0);
	    
	  } // end for i,j,k
    
    } // end THREE_D

  } // MHDRunBase::init_mhd_Rayleigh_Taylor

  // =======================================================
  // =======================================================
  /**
   * The 3D forcing MHD turbulence problem.
   *
   */
  void MHDRunBase::init_mhd_turbulence()
  {

    if (dimType == TWO_D) {
      
      std::cerr << "Turbulence problem is not available in 2D...." << std::endl;
      
    } else { // THREE_D
      
      // magnetic field initial conditions
      real_t Bx0 = configMap.getFloat("turbulence", "bx",  1e-8);
      real_t By0 = configMap.getFloat("turbulence", "by",  1e-8);
      real_t Bz0 = configMap.getFloat("turbulence", "bz",  1e-8);
      
      // beta plasma is ratio between plasma pressure (p=n*k_B*T)
      // and magnetic pressure (P_mag=B^2/2*mu_0)
      real_t beta = configMap.getFloat("turbulence", "beta",  0.0);

      
      if (beta>0) {
	// use that beta to initialize B along x
	double cIso2 = _gParams.cIso * _gParams.cIso;
	double d0    = configMap.getFloat("turbulence", "density",  1.0);
	Bx0 = sqrt(2*cIso2*d0/beta);
	By0 = 0.0;
	Bz0 = 0.0;
	
	if (cIso2 <= 0.0) { // non-isothermal simulation
	  Bx0 = configMap.getFloat("turbulence", "Bx0",  2.0*d0/beta);
	}

      } // end beta > 0

      // call hydro initialization routine
      init_hydro_turbulence();
      
      // initialize magnetic field
      for (int k=ghostWidth; k<ksize-ghostWidth; k++)
	for (int j=ghostWidth; j<jsize-ghostWidth; j++)
	  for (int i=ghostWidth; i<isize-ghostWidth; i++) {
	    h_U(i,j,k,IBX) = Bx0;
	    h_U(i,j,k,IBY) = By0;
	    h_U(i,j,k,IBZ) = Bz0;

	    // update energy
	    h_U(i,j,k,IP) += 0.5 * (Bx0*Bx0 + By0*By0 + Bz0*Bz0);

	  } // end for i,j,k

    } // end THREE_D

  } // MHDRunBase::init_mhd_turbulence

  // =======================================================
  // =======================================================
  /**
   *
   * The 3D forcing MHD turbulence problem using Ornstein-Uhlenbeck forcing.
   *
   */
  void MHDRunBase::init_mhd_turbulence_Ornstein_Uhlenbeck()
  {

    if (dimType == TWO_D) {
      
      std::cerr << "Turbulence problem is not available in 2D...." << std::endl;
      
    } else { // THREE_D
      
      // magnetic field initial conditions
      real_t Bx0 = configMap.getFloat("turbulence-Ornstein-Uhlenbeck", "bx",  1e-8);
      real_t By0 = configMap.getFloat("turbulence-Ornstein-Uhlenbeck", "by",  1e-8);
      real_t Bz0 = configMap.getFloat("turbulence-Ornstein-Uhlenbeck", "bz",  1e-8);
      
      // beta plasma is ratio between plasma pressure (p=n*k_B*T)
      // and magnetic pressure (P_mag=B^2/2*mu_0)
      real_t beta = configMap.getFloat("turbulence-Ornstein-Uhlenbeck", "beta",  0.0);

      
      if (beta>0) {
	// use that beta to initialize B along x
	double cIso2 = _gParams.cIso * _gParams.cIso;
	double d0    = configMap.getFloat("turbulence-Ornstein-Uhlenbeck", "density",  1.0);
	Bx0 = sqrt(2*cIso2*d0/beta);
	By0 = 0.0;
	Bz0 = 0.0;
	
	if (cIso2 <= 0.0) { // non-isothermal simulation
	  Bx0 = configMap.getFloat("turbulence-Ornstein-Uhlenbeck", "Bx0",  2.0*d0/beta);
	}

      }

      // call hydro initialization routine
      init_hydro_turbulence_Ornstein_Uhlenbeck();
      
      // initialize magnetic field
      for (int k=ghostWidth; k<ksize-ghostWidth; k++)
	for (int j=ghostWidth; j<jsize-ghostWidth; j++)
	  for (int i=ghostWidth; i<isize-ghostWidth; i++) {

	    h_U(i,j,k,IBX) = Bx0;
	    h_U(i,j,k,IBY) = By0;
	    h_U(i,j,k,IBZ) = Bz0;

	    // update energy
	    h_U(i,j,k,IP) += 0.5 * (Bx0*Bx0 + By0*By0 + Bz0*Bz0);

	  } // end for i,j,k

    } // end THREE_D

  } // MHDRunBase::init_mhd_turbulence_Ornstein_Uhlenbeck

  // =======================================================
  // =======================================================
  void MHDRunBase::init_mhd_mri_grav_field()
  {

    // initialize gravity field
    if (gravityEnabled) {

      double phi[2][THREE_D] = {0.0};
      h_gravity.reset();
      
      bool smoothGravity = configMap.getBool ("MRI", "smoothGravity", false);
      double zFloor      = configMap.getFloat("MRI", "zFloor"       , 5.0);
      
      real_t &Omega0     = _gParams.Omega0;
      //real_t &cIso       = _gParams.cIso;
      //double H           = cIso/Omega0;
      
      for (int k=0; k<ksize; k++) {
	real_t zPos = _gParams.zMin + dz/2 + (k-ghostWidth)*dz;
	
	for (int j=0; j<jsize; j++) {
	  //real_t yPos = _gParams.yMin + dy/2 + (j-ghostWidth)*dy;
	  
	  for (int i=0; i<isize; i++) {
	    //real_t xPos = _gParams.xMin + dx/2 + (i-ghostWidth)*dx;
	    
	    phi[0][IZ] = HALF_F*Omega0*Omega0*(zPos-dz)*(zPos-dz);
	    phi[1][IZ] = HALF_F*Omega0*Omega0*(zPos+dz)*(zPos+dz);
	    
	    if (smoothGravity) {
              if ( (zPos-dz)>zFloor ) phi[0][IZ] = HALF_F*Omega0*Omega0*zFloor*zFloor;
              if ( (zPos+dz)>zFloor ) phi[1][IZ] = HALF_F*Omega0*Omega0*zFloor*zFloor;
	    }
	    
	    h_gravity(i,j,k,IX) = - HALF_F * ( phi[1][IX] - phi[0][IX] ) / dx;
	    h_gravity(i,j,k,IY) = - HALF_F * ( phi[1][IY] - phi[0][IY] ) / dy;
	    h_gravity(i,j,k,IZ) = - HALF_F * ( phi[1][IZ] - phi[0][IZ] ) / dz;
	    
	  } // end for i
	} // end for j
      } // end for k

      // need to upload gravity field on GPU
#ifdef __CUDACC__
    d_gravity.copyFromHost(h_gravity);
#endif

    } // end gravityEnabled

  } // MHDRunBase::init_mhd_mri_grav_field
  
  // =======================================================
  // =======================================================
  void MHDRunBase::restart_run_extra_work()
  {

    if ( (!problem.compare("MRI") ||
	  !problem.compare("Mri") ||
	  !problem.compare("mri") ) and gravityEnabled) {

      // we need to re-generate gravity field
      init_mhd_mri_grav_field();

    } // end extra stuff for restarting a stratified mri run

  } // MHDRunBase::restart_run_extra_work

  // =======================================================
  // =======================================================
  /*
   * setup history, choose which history method will be called
   */
  void MHDRunBase::setupHistory()
  {

    // history enabled ?
    bool historyEnabled = configMap.getBool("history","enabled",false);

    if (historyEnabled) {
      
      if (!problem.compare("InertialWave") ||
	  !problem.compare("inertialwave") ||
	  !problem.compare("Inertial-Wave") ||
	  !problem.compare("inertial-wave") ||
	  !problem.compare("Inertialwave")) {
	
	history_method = &MHDRunBase::history_inertial_wave;
	
      } else if (!problem.compare("Orszag-Tang") || 
		 !problem.compare("OrszagTang") ) {

	history_method = &MHDRunBase::history_default;

      } else if ( !problem.compare("turbulence") ||
		  !problem.compare("turbulence-Ornstein-Uhlenbeck")) {

	history_method = &MHDRunBase::history_turbulence;

      } else if (!problem.compare("MRI") ||
		 !problem.compare("Mri") ||
		 !problem.compare("mri")) {

	history_method = &MHDRunBase::history_mri;
      
      } else {
	
	history_method = &MHDRunBase::history_empty;
	
      }
      
    } else { // history disabled

      history_method = &MHDRunBase::history_empty;
      
    }

  } // MHDRunBase::setupHistory

  // =======================================================
  // =======================================================
  /*
   * call history.
   */
  void MHDRunBase::history(int nStep, real_t dt)
  {
    // call the actual history method
    ((*this).*history_method)(nStep,dt);

  } // MHDRunBase::history

  // =======================================================
  // =======================================================
  /*
   * Default history, do nothing.
   */
  void MHDRunBase::history_empty(int nStep, real_t dt)
  {
    (void) nStep;
    (void) dt;

  } // MHDRunBase::history_empty

  // =======================================================
  // =======================================================
  /*
   * Default history.
   * only compute total mass (should be constant in time)
   * and divB (should be constant as small as possible)
   */
  void MHDRunBase::history_default(int nStep, real_t dt)
  {
    (void) nStep;
    (void) dt;

    std::cout << "History at time " << totalTime << "\n";
      
    // history file name
    std::string fileName = configMap.getString("history",
					       "filename", 
					       "history.txt");
    // get output prefix / outputDir
    std::string outputDir    = configMap.getString("output", "outputDir", "./");
    std::string outputPrefix = configMap.getString("output", "outputPrefix", "output");
    
    // build full path filename
    fileName = outputDir + "/" + outputPrefix + "_" + fileName;
    
    // open history file
    std::ofstream histo;
    histo.open (fileName.c_str(), std::ios::out | std::ios::app | std::ios::ate); 
    
    // if this is the first time we call history, print header
    if (totalTime <= 0) {
      histo << "# history " << current_date() << std::endl;
      
      bool restartEnabled = configMap.getBool("run","restart",false);
      if (restartEnabled)
	histo << "# history : this is a restart run\n";
      
      // write header (which variables are dumped)
      histo << "# totalTime dt mass divB\n";
      
    } // end print header
    
    // make sure Device data are copied back onto Host memory
    // which data to save ?
    copyGpuToCpu(nStep);
    
    HostArray<real_t> &U = getDataHost(nStep);
    
    // compute total mass using reduction algorithm
    double mass = 0.0;
    double divB = 0.0;
    
    if (dimType == TWO_D) {

	for (int j=ghostWidth; j<jsize-ghostWidth; j++) {
	  for (int i=ghostWidth; i<isize-ghostWidth; i++) {
	    
	    mass += U(i,j,ID);
	    
	    divB +=  
	      ( U(i+1,j  ,IBX) - U(i,j,IBX) ) / dx + 
	      ( U(i  ,j+1,IBY) - U(i,j,IBY) ) / dy;

	  } // end for i
	} // end for j

	double dTau = dx*dy/
	  (_gParams.xMax- _gParams.xMin)/
	  (_gParams.yMax- _gParams.yMin);
	
	mass = mass*dTau;

    } else { // THREE_D

      for (int k=ghostWidth; k<ksize-ghostWidth; k++) {
	for (int j=ghostWidth; j<jsize-ghostWidth; j++) {
	  for (int i=ghostWidth; i<isize-ghostWidth; i++) {
	    
	    mass += U(i,j,k,ID);
	    
	    divB +=  
	      ( U(i+1,j  ,k  ,IBX) - U(i,j,k,IBX) ) / dx + 
	      ( U(i  ,j+1,k  ,IBY) - U(i,j,k,IBY) ) / dy + 
	      ( U(i  ,j  ,k+1,IBZ) - U(i,j,k,IBZ) ) / dz;

	  } // end for i
	} // end for j
      } // end for k

      double dTau = dx*dy*dz/
	(_gParams.xMax- _gParams.xMin)/
	(_gParams.yMax- _gParams.yMin)/
	(_gParams.zMax- _gParams.zMin);
      
      mass = mass*dTau;

    } // end THREE_D

    histo << totalTime << "\t" << dt   << "\t" 
	  << mass      << "\t" << divB << "\n";
    
    histo.close();
    
  } // MHDRunBase::history_default

  // =======================================================
  // =======================================================
  /*
   * history for inertial wave problem.
   */
  void MHDRunBase::history_inertial_wave(int nStep, real_t dt)
  {

    std::cout << "History for inertial wave problem\n";

    // history file name
    std::string fileName = configMap.getString("history",
					       "filename", 
					       "history.txt");
    // get output prefix / outputDir
    std::string outputDir    = configMap.getString("output", "outputDir", "./");
    std::string outputPrefix = configMap.getString("output", "outputPrefix", "output");

    // build full path filename
    fileName = outputDir + "/" + outputPrefix + "_" + fileName;

    // open history file
    std::ofstream histo;
    histo.open (fileName.c_str(), std::ios::out | std::ios::app | std::ios::ate); 
    
    // if this is the first time we call history, print header
    if (totalTime <= 0) {
      histo << "# history " << current_date() << std::endl;
      
      bool restartEnabled = configMap.getBool("run","restart",false);
      if (restartEnabled)
	histo << "# history : this is a restart run\n";
    }

    // make sure Device data are copied back onto Host memory
    // which data to save ?
    copyGpuToCpu(nStep);
    
    HostArray<real_t> &U = getDataHost(nStep);

    int iPos = ghostWidth+nx/2;
    int kPos = ghostWidth; // first cell inside domain after ghost
    real_t rho, dvx, dvy;

    if (dimType==TWO_D) {
      rho = U(iPos,kPos,ID);
      dvx = U(iPos,kPos,IU)/rho;
      dvy = U(iPos,kPos,IV)/rho;
    } else { // THREE_D
      rho = U(iPos,1,kPos,ID);
      dvx = U(iPos,1,kPos,IU)/rho;
      dvy = U(iPos,1,kPos,IV)/rho;
    }

    histo << totalTime << "" << dt << " " << rho << " "
	  << fmt(dvx/_gParams.cIso) << " " 
	  << fmt(dvy/_gParams.cIso) << "\n";
    
    histo.close();

  } // MHDRunBase::history_inertial_wave

  // =======================================================
  // =======================================================
  /*
   * history for MRI problem.
   */
  void MHDRunBase::history_mri(int nStep, real_t dt)
  {

    std::cout << "History for MRI problem at time " << totalTime << "\n";

    if (dimType == TWO_D) {
      // don't do anything
    } else {
      
      // history file name
      std::string fileName = configMap.getString("history",
						 "filename", 
						 "history.txt");
      // get output prefix / outputDir
      std::string outputDir    = configMap.getString("output", "outputDir", "./");
      std::string outputPrefix = configMap.getString("output", "outputPrefix", "output");
      
      // build full path filename
      fileName = outputDir + "/" + outputPrefix + "_" + fileName;
      
      // open history file
      std::ofstream histo;
      histo.open (fileName.c_str(), std::ios::out | std::ios::app | std::ios::ate); 
      
      // if this is the first time we call history, print header
      if (totalTime <= 0) {
	histo << "# history " << current_date() << std::endl;
	
	bool restartEnabled = configMap.getBool("run","restart",false);
	if (restartEnabled)
	  histo << "# history : this is a restart run\n";
	
	// write header (which variables are dumped)
	histo << "# totalTime dt mass maxwell reynolds maxwell+reynolds magp ";
	histo << "mean_Bx mean_By mean_Bz divB\n";

      } // end print header

      // make sure Device data are copied back onto Host memory
      // which data to save ?
      copyGpuToCpu(nStep);
      
      HostArray<real_t> &U = getDataHost(nStep);
      
      double mass = 0.0, magp = 0.0, maxwell = 0.0;
      double mean_Bx = 0.0, mean_By = 0.0, mean_Bz = 0.0;
      
      // do a local reduction
      for (int k=ghostWidth; k<ksize-ghostWidth; k++) {
	for (int j=ghostWidth; j<jsize-ghostWidth; j++) {
	  for (int i=ghostWidth; i<isize-ghostWidth; i++) {
	    
	    mass += U(i,j,k,ID);
	    magp += 0.25*SQR( U(i,j,k,IBX)+U(i+1,j  ,k  ,IBX) );
	    magp += 0.25*SQR( U(i,j,k,IBY)+U(i  ,j+1,k  ,IBY) );
	    magp += 0.25*SQR( U(i,j,k,IBZ)+U(i  ,j  ,k+1,IBZ) );
	    
	    maxwell -= 0.25 *
	      ( U(i,j,k,IBX) + U(i+1,j  ,k,IBX) ) * 
	      ( U(i,j,k,IBY) + U(i  ,j+1,k,IBY) );
	    
	    mean_Bx += U(i,j,k,IBX);
	    mean_By += U(i,j,k,IBY);
	    mean_Bz += U(i,j,k,IBZ);
	    
	  } // end for i
	} // end for j
      } // end for k
      
      double dTau = dx*dy*dz/
	(_gParams.xMax- _gParams.xMin)/
	(_gParams.yMax- _gParams.yMin)/
	(_gParams.zMax- _gParams.zMin);
      
      magp    = magp*dTau/2.;
      mass    = mass*dTau;
      maxwell = maxwell*dTau;
      mean_Bx = mean_Bx*dTau;
      mean_By = mean_By*dTau;
      mean_Bz = mean_Bz*dTau;
      
     
      /* 
       * compute Y-Z averages
       */
      HostArray<real_t> localMean; 
      // 3 field: rho, rhovx/rho, rhovy/rho
      localMean.allocate( isize, 3 ); 
      // reset 
      memset( localMean.data(), 0, localMean.sizeBytes() );

      for (int k=ghostWidth; k<ksize-ghostWidth; k++) {
	for (int j=ghostWidth; j<jsize-ghostWidth; j++) {
	  for (int i=0; i<isize; i++) {

	    localMean(i,0) += U(i,j,k,ID);
	    localMean(i,1) += U(i,j,k,IU)/U(i,j,k,ID);
	    localMean(i,2) += U(i,j,k,IV)/U(i,j,k,ID);

	  } // end for i
	} // end for j
      } // end for k

      for (int i=0; i<isize; i++) {
	localMean(i,0) /= (ny*nz);
  	localMean(i,1) /= (ny*nz);
 	localMean(i,2) /= (ny*nz);
      }


      /*
       * compute Reynolds number and divB
       */
      double reynolds = 0.0;
      double divB = 0.0;
      for (int k=ghostWidth; k<ksize-ghostWidth; k++) {
	for (int j=ghostWidth; j<jsize-ghostWidth; j++) {
	  for (int i=ghostWidth; i<isize-ghostWidth; i++) {
	    
	    reynolds += U(i,j,k,ID) * dTau *
	      ( U(i,j,k,IU) / U(i,j,k,ID) - localMean(i,1) ) *
	      ( U(i,j,k,IV) / U(i,j,k,ID) - localMean(i,2) );
	    
	    divB +=  
	      ( U(i+1,j  ,k  ,IBX) - U(i,j,k,IBX) ) / dx + 
	      ( U(i  ,j+1,k  ,IBY) - U(i,j,k,IBY) ) / dy + 
	      ( U(i  ,j  ,k+1,IBZ) - U(i,j,k,IBZ) ) / dz;
	    
	  } // end for i
	} // end for j
      } // end for k
      
      histo << totalTime << "\t" << dt               << "\t" 
	    << mass      << "\t" << maxwell          << "\t" 
	    << reynolds  << "\t" << maxwell+reynolds << "\t"
	    << magp      << "\t" << mean_Bx          << "\t"
	    << mean_By   << "\t" << mean_Bz          << "\t"
	    << divB      << "\n";
		   
      histo.close();

    } // end THREE_D

  } // MHDRunBase::history_mri

  // =======================================================
  // =======================================================
  /*
   * history for turbulence problem.
   */
  void MHDRunBase::history_turbulence(int nStep, real_t dt)
  {

    std::cout << "History for turbulence problem at time " << totalTime << "\n";

    if (dimType == TWO_D) {

      // don't do anything, this problem is not available in 2D

    } else {
      
      // history file name
      std::string fileName = configMap.getString("history",
						 "filename", 
						 "history.txt");
      // get output prefix / outputDir
      std::string outputDir    = configMap.getString("output", "outputDir", "./");
      std::string outputPrefix = configMap.getString("output", "outputPrefix", "output");
      
      // build full path filename
      fileName = outputDir + "/" + outputPrefix + "_" + fileName;
      
      // open history file
      std::ofstream histo;
      histo.open (fileName.c_str(), std::ios::out | std::ios::app | std::ios::ate); 
      
      // if this is the first time we call history, print header
      if (totalTime <= 0) {
	histo << "# history " << current_date() << std::endl;
	
	bool restartEnabled = configMap.getBool("run","restart",false);
	if (restartEnabled)
	  histo << "# history : this is a restart run\n";
	
	// write header (which variables are dumped)
	// Ma_s is the sonic Mach number Ma_s = v_rms/c_s
	// Ma_alfven is the alfvenic Mach number (v_rms / v_0A) where 
	// v_rms = sqrt(<v^2) and v_0A = B_0/sqrt(4 pi rho_0) 
	// helicity is the mean value < (rho_v) . (B/sqrt(rho)) >
	histo << "# totalTime dt mass divB eKin eMag helicity mean_rho mean_B mean_Bx mean_By mean_Bz mean_rhovx mean_rhovy mean_rhovz Ma_s Ma_alfven coef_x coef_y coef_z\n";

      } // end print header

      // make sure Device data are copied back onto Host memory
      // which data to save ?
      copyGpuToCpu(nStep);
      
      HostArray<real_t> &U = getDataHost(nStep);

      const double pi = 2*asin(1.0);
      
      double mass       = 0.0, eKin       = 0.0, eMag       = 0.0;
      double helicity   = 0.0;
      double mean_Bx    = 0.0, mean_By    = 0.0, mean_Bz    = 0.0;
      double mean_rhovx = 0.0, mean_rhovy = 0.0, mean_rhovz = 0.0;
      double mean_v2    = 0.0, mean_rho   = 0.0;
      
      // monitor high k DFT coefficients of rho
      int kfft = nx-3;
      double coef_x_re = 0.0, coef_x_im = 0.0;
      double coef_y_re = 0.0, coef_y_im = 0.0;
      double coef_z_re = 0.0, coef_z_im = 0.0;

      // do a local reduction
      for (int k=ghostWidth; k<ksize-ghostWidth; k++) {
	for (int j=ghostWidth; j<jsize-ghostWidth; j++) {
	  for (int i=ghostWidth; i<isize-ghostWidth; i++) {

	    real_t rho = U(i,j,k,ID);
	    real_t bx  = U(i,j,k,IBX);
	    mass += rho;

	    eKin += SQR( U(i,j,k,IU) ) / rho;
	    eKin += SQR( U(i,j,k,IV) ) / rho;
	    eKin += SQR( U(i,j,k,IW) ) / rho;
	    
	    mean_v2 += SQR( U(i,j,k,IU)/rho );
	    mean_v2 += SQR( U(i,j,k,IV)/rho );
	    mean_v2 += SQR( U(i,j,k,IW)/rho );

	    eMag += SQR( U(i,j,k,IBX) );
	    eMag += SQR( U(i,j,k,IBY) );
	    eMag += SQR( U(i,j,k,IBZ) );

	    helicity += U(i,j,k,IU)*U(i,j,k,IBX)/sqrt(rho);
	    helicity += U(i,j,k,IV)*U(i,j,k,IBY)/sqrt(rho);
  	    helicity += U(i,j,k,IW)*U(i,j,k,IBZ)/sqrt(rho);
	    
	    mean_Bx += U(i,j,k,IBX);
	    mean_By += U(i,j,k,IBY);
	    mean_Bz += U(i,j,k,IBZ);
	    
	    mean_rhovx += U(i,j,k,IU);
	    mean_rhovy += U(i,j,k,IV);
	    mean_rhovz += U(i,j,k,IW);

	    mean_rho += rho;
	    
	    coef_x_re += bx*cos(2*pi*kfft*i/nx);
	    coef_x_im += bx*sin(2*pi*kfft*i/nx);
	    coef_y_re += bx*cos(2*pi*kfft*j/ny);
	    coef_y_im += bx*sin(2*pi*kfft*j/ny);
	    coef_z_re += bx*cos(2*pi*kfft*k/nz);
	    coef_z_im += bx*sin(2*pi*kfft*k/nz);

	  } // end for i
	} // end for j
      } // end for k

      
      
      double dTau = dx*dy*dz/
	(_gParams.xMax- _gParams.xMin)/
	(_gParams.yMax- _gParams.yMin)/
	(_gParams.zMax- _gParams.zMin);
      
      mass    = mass*dTau;

      eKin = eKin*dTau;
      eMag = eMag*dTau;
      helicity *= dTau;

      mean_Bx = mean_Bx*dTau;
      mean_By = mean_By*dTau;
      mean_Bz = mean_Bz*dTau;

      double mean_B = sqrt( SQR(mean_Bx) + SQR(mean_By) + SQR(mean_Bz) );

      mean_rhovx = mean_rhovx*dTau;
      mean_rhovy = mean_rhovy*dTau;
      mean_rhovz = mean_rhovz*dTau;

      mean_v2  = mean_v2*dTau;
      
      mean_rho = mean_rho*dTau;

      double coef_x = sqrt( SQR(coef_x_re) + SQR(coef_x_im) );
      coef_x *= dTau;
      double coef_y = sqrt( SQR(coef_y_re) + SQR(coef_y_im) );
      coef_y *= dTau;
      double coef_z = sqrt( SQR(coef_z_re) + SQR(coef_z_im) );
      coef_z *= dTau;

      /*
       * compute divB
       */
      double divB = 0.0;
      for (int k=ghostWidth; k<ksize-ghostWidth; k++) {
	for (int j=ghostWidth; j<jsize-ghostWidth; j++) {
	  for (int i=ghostWidth; i<isize-ghostWidth; i++) {
	    
	    divB +=  
	      ( U(i+1,j  ,k  ,IBX) - U(i,j,k,IBX) ) / dx + 
	      ( U(i  ,j+1,k  ,IBY) - U(i,j,k,IBY) ) / dy + 
	      ( U(i  ,j  ,k+1,IBZ) - U(i,j,k,IBZ) ) / dz;
	    
	  } // end for i
	} // end for j
      } // end for k
      
      double Ma_alfven = sqrt(mean_v2)/(mean_B/sqrt(4*pi*mean_rho));
      real_t &cIso = _gParams.cIso;
      double Ma_s = sqrt(mean_v2)/cIso; 

      histo << totalTime  << "\t" << dt         << "\t" 
	    << mass       << "\t" << divB       << "\t" 
	    << eKin       << "\t" << eMag       << "\t"
	    << helicity   << "\t"
	    << mean_rho   << "\t" << mean_B     << "\t"
	    << mean_Bx    << "\t" << mean_By    << "\t" << mean_Bz    << "\t"
	    << mean_rhovx << "\t" << mean_rhovy << "\t" << mean_rhovz << "\t"
	    << Ma_s       << "\t" << Ma_alfven  << "\t"
	    << coef_x     << "\t" << coef_y     << "\t" << coef_z     << "\n";
		   
      histo.close();

    } // end THREE_D

    bool structureFunctionsEnabled = configMap.getBool("structureFunctions","enabled",false);
    if ( structureFunctionsEnabled ) {
      HostArray<real_t> &U = getDataHost(nStep);
      structure_functions_mhd(nStep,configMap,_gParams,U);
    }
    
  } // MHDRunBase::history_turbulence

} // namespace hydroSimu
