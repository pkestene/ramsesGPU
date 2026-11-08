/**
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
 * \file kernel_godunov_unsplit_3d.cuh
 * \brief Define 3d godunov kernel.
 */
#ifndef KERNEL_GODUNOV_UNSPLIT_3D_CUH_
#define KERNEL_GODUNOV_UNSPLIT_3D_CUH_

#include "real_type.h"
#include "constants.h"

// 3D-kernel block dimensions
#ifdef USE_DOUBLE
// newer tested
#  define UNSPLIT_BLOCK_DIMX_3D 16
#  define UNSPLIT_BLOCK_INNER_DIMX_3D (UNSPLIT_BLOCK_DIMX_3D - 4)
#  define UNSPLIT_BLOCK_DIMY_3D 12
#  define UNSPLIT_BLOCK_INNER_DIMY_3D (UNSPLIT_BLOCK_DIMY_3D - 4)
#else // simple precision
#  define UNSPLIT_BLOCK_DIMX_3D 16
#  define UNSPLIT_BLOCK_INNER_DIMX_3D (UNSPLIT_BLOCK_DIMX_3D - 4)
#  define UNSPLIT_BLOCK_DIMY_3D 12
#  define UNSPLIT_BLOCK_INNER_DIMY_3D (UNSPLIT_BLOCK_DIMY_3D - 4)
#endif // USE_DOUBLE

/**
 * Unsplit Godunov kernel for 3D data (in place computation).
 *
 * The main difference with the 2D kernel is that we perform a sweep
 * along the 3rd direction.
 *
 * In place computation is probably not efficient, we need to store to
 * much data (4 planes) into shared memory and then to do not have
 * enough shared memory
 *
 * \note Note that when gravity is enabled, only predictor step is performed here;
 * the source term computation must be done outside !!!
 *
 */
__global__ void
kernel_godunov_unsplit_3d(const real_t * __restrict__ Uin,
                          real_t * Uout,
                          int      pitch,
                          int      imax,
                          int      jmax,
                          int      kmax,
                          real_t   dtdx,
                          real_t   dt,
                          bool     gravityEnabled)
{

  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;

  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;

  const int i = __mul24(bx, UNSPLIT_BLOCK_INNER_DIMX_3D) + tx;
  const int j = __mul24(by, UNSPLIT_BLOCK_INNER_DIMY_3D) + ty;

  // 3D array size
  const int arraySize = pitch * jmax * kmax;

  // we always store in shared 4 consecutive XY-plans
  __shared__ real_t q[4][UNSPLIT_BLOCK_DIMX_3D][UNSPLIT_BLOCK_DIMY_3D][NVAR_3D];

  real_t * gravin = gParams.arrayList[A_GRAV];

  // index to address the 4 plans of data
  int low, mid, current, top, tmp;
  low = 0;
  mid = 1;
  current = 2;
  top = 3;

  // intermediate variables used to build qleft, qright as input to
  // Riemann problem
  real_t qm[THREE_D][NVAR_3D];
  real_t qm1[NVAR_3D];
  real_t qm2[NVAR_3D];
  real_t qm3[NVAR_3D];

  real_t qp[THREE_D][NVAR_3D];
  real_t qp0[THREE_D][NVAR_3D];

  // conservative variables
  real_t uIn[NVAR_3D];
  real_t uOut[NVAR_3D];
  real_t c[4];

  /*
   * initialize q with the first 4 plans
   */
  for (int k = 0, elemOffset = i + pitch * j; k < 4; ++k, elemOffset += (pitch * jmax))
  {

    if (i >= 0 and i < imax and j >= 0 and j < jmax)
    {

      // Gather conservative variables
      int offset = elemOffset;
      uIn[ID] = Uin[offset];
      offset += arraySize;
      uIn[IP] = Uin[offset];
      offset += arraySize;
      uIn[IU] = Uin[offset];
      offset += arraySize;
      uIn[IV] = Uin[offset];
      offset += arraySize;
      uIn[IW] = Uin[offset];

      // Convert to primitive variables
      constoprim_3D(uIn, q[k][tx][ty], c[k]);
    }
  } // end loading the first 4 plans
  __syncthreads();

  /*
   * loop over all X-Y-planes starting at z=2 as the current plane.
   * Note that elemOffset is used in the update stage
   */
  for (int k = 2, elemOffset = i + pitch * (j + jmax * 2); k < kmax - 1;
       ++k, elemOffset += (pitch * jmax))
  {

    if (i > 1 and i < imax - 1 and tx > 1 and tx < UNSPLIT_BLOCK_DIMX_3D - 1 and j > 1 and
        j < jmax - 1 and ty > 1 and ty < UNSPLIT_BLOCK_DIMY_3D - 1)
    {
      // qNeighbors is used for trace computations
      real_t qNeighbors[2 * THREE_D][NVAR_3D];

      // Characteristic tracing : compute qp0
      for (int iVar = 0; iVar < NVAR_3D; iVar++)
      {
        qNeighbors[0][iVar] = q[current][tx + 1][ty][iVar];
        qNeighbors[1][iVar] = q[current][tx - 1][ty][iVar];
        qNeighbors[2][iVar] = q[current][tx][ty + 1][iVar];
        qNeighbors[3][iVar] = q[current][tx][ty - 1][iVar];
        qNeighbors[4][iVar] = q[top][tx][ty][iVar];
        qNeighbors[5][iVar] = q[mid][tx][ty][iVar];
      }
      trace_unsplit<THREE_D, NVAR_3D>(q[current][tx][ty], qNeighbors, c[current], dtdx, qm, qp0);

      // gravity predictor on velocity components of qp0's
      if (gravityEnabled)
      {
        qp0[0][IU] += HALF_F * dt * gravin[elemOffset + IX * arraySize];
        qp0[0][IV] += HALF_F * dt * gravin[elemOffset + IY * arraySize];
        qp0[0][IW] += HALF_F * dt * gravin[elemOffset + IZ * arraySize];

        qp0[1][IU] += HALF_F * dt * gravin[elemOffset + IX * arraySize];
        qp0[1][IV] += HALF_F * dt * gravin[elemOffset + IY * arraySize];
        qp0[1][IW] += HALF_F * dt * gravin[elemOffset + IZ * arraySize];

        qp0[2][IU] += HALF_F * dt * gravin[elemOffset + IX * arraySize];
        qp0[2][IV] += HALF_F * dt * gravin[elemOffset + IY * arraySize];
        qp0[2][IW] += HALF_F * dt * gravin[elemOffset + IZ * arraySize];
      }

      // Characteristic tracing : compute qm_x[1] (shift x -1)
      for (int iVar = 0; iVar < NVAR_3D; iVar++)
      {
        qNeighbors[0][iVar] = q[current][tx][ty][iVar];
        qNeighbors[1][iVar] = q[current][tx - 2][ty][iVar];
        qNeighbors[2][iVar] = q[current][tx - 1][ty + 1][iVar];
        qNeighbors[3][iVar] = q[current][tx - 1][ty - 1][iVar];
        qNeighbors[4][iVar] = q[top][tx - 1][ty][iVar];
        qNeighbors[5][iVar] = q[mid][tx - 1][ty][iVar];
      }
      trace_unsplit<THREE_D, NVAR_3D>(q[current][tx - 1][ty], qNeighbors, c[current], dtdx, qm, qp);
      for (int iVar = 0; iVar < NVAR_3D; iVar++)
      {
        qm1[iVar] = qm[0][iVar];
      }

      // gravity predictor on velocity components of qm1
      if (gravityEnabled)
      {
        qm1[IU] += HALF_F * dt * gravin[elemOffset - 1 + IX * arraySize];
        qm1[IV] += HALF_F * dt * gravin[elemOffset - 1 + IY * arraySize];
        qm1[IW] += HALF_F * dt * gravin[elemOffset - 1 + IZ * arraySize];
      }

      // Characteristic tracing : compute qm_y[2] (shift y -1)
      for (int iVar = 0; iVar < NVAR_3D; iVar++)
      {
        qNeighbors[0][iVar] = q[current][tx + 1][ty - 1][iVar];
        qNeighbors[1][iVar] = q[current][tx - 1][ty - 1][iVar];
        qNeighbors[2][iVar] = q[current][tx][ty][iVar];
        qNeighbors[3][iVar] = q[current][tx][ty - 2][iVar];
        qNeighbors[4][iVar] = q[top][tx][ty - 1][iVar];
        qNeighbors[5][iVar] = q[mid][tx][ty - 1][iVar];
      }
      trace_unsplit<THREE_D, NVAR_3D>(q[current][tx][ty - 1], qNeighbors, c[current], dtdx, qm, qp);
      for (int iVar = 0; iVar < NVAR_3D; iVar++)
      {
        qm2[iVar] = qm[1][iVar];
      }

      // gravity predictor on velocity components of qm2
      if (gravityEnabled)
      {
        qm2[IU] += HALF_F * dt * gravin[elemOffset - pitch + IX * arraySize];
        qm2[IV] += HALF_F * dt * gravin[elemOffset - pitch + IY * arraySize];
        qm2[IW] += HALF_F * dt * gravin[elemOffset - pitch + IZ * arraySize];
      }

      // Characteristic tracing : compute qm_z[3] (shift z -1)
      for (int iVar = 0; iVar < NVAR_3D; iVar++)
      {
        qNeighbors[0][iVar] = q[mid][tx + 1][ty][iVar];
        qNeighbors[1][iVar] = q[mid][tx - 1][ty][iVar];
        qNeighbors[2][iVar] = q[mid][tx][ty + 1][iVar];
        qNeighbors[3][iVar] = q[mid][tx][ty - 1][iVar];
        qNeighbors[4][iVar] = q[current][tx][ty][iVar];
        qNeighbors[5][iVar] = q[low][tx][ty][iVar];
      }
      trace_unsplit<THREE_D, NVAR_3D>(q[mid][tx][ty], qNeighbors, c[mid], dtdx, qm, qp);
      for (int iVar = 0; iVar < NVAR_3D; iVar++)
      {
        qm3[iVar] = qm[2][iVar];
      }

      // gravity predictor on velocity components of qm3
      if (gravityEnabled)
      {
        qm3[IU] += HALF_F * dt * gravin[elemOffset - pitch * jmax + IX * arraySize];
        qm3[IV] += HALF_F * dt * gravin[elemOffset - pitch * jmax + IY * arraySize];
        qm3[IW] += HALF_F * dt * gravin[elemOffset - pitch * jmax + IZ * arraySize];
      }
    }
    // end trace/slope computations, we have all we need to
    // compute qleft/qright now
    __syncthreads();

    /*
     * Now, compute fluxes !
     */
    // a nice trick: reuse q[low] to store fluxes, as we don't need then anymore
    // also remember that q[low] will become q[top] after the flux
    // computations and hydro update
    real_t(&flux)[UNSPLIT_BLOCK_DIMX_3D][UNSPLIT_BLOCK_DIMY_3D][NVAR_3D] = q[low];

    // solve Riemann problems at X interfaces
    flux[tx][ty][ID] = ZERO_F;
    flux[tx][ty][IP] = ZERO_F;
    flux[tx][ty][IU] = ZERO_F;
    flux[tx][ty][IV] = ZERO_F;
    flux[tx][ty][IW] = ZERO_F;
    __syncthreads();
    if (i >= 2 and i < imax - 1 and tx > 1 and tx < UNSPLIT_BLOCK_DIMX_3D - 1 and j >= 2 and
        j < jmax - 1 and ty > 1 and ty < UNSPLIT_BLOCK_DIMY_3D - 1)
    {
      real_t qgdnv[NVAR_3D];

      // Solve Riemann problem at X-interfaces and compute fluxes
      real_t(&qleft_x)[NVAR_3D] = qm1;
      real_t(&qright_x)[NVAR_3D] = qp0[0];
      riemann<NVAR_3D>(qleft_x, qright_x, qgdnv, flux[tx][ty]);
    } // end solve Riemann at X interfaces
    __syncthreads();

    // update hydro along X
    if (i >= 2 and i < imax - 1 and tx > 1 and tx < UNSPLIT_BLOCK_DIMX_3D - 2 and j >= 2 and
        j < jmax - 1 and ty > 1 and ty < UNSPLIT_BLOCK_DIMY_3D - 2)
    {
      // re-read input state into uOut which will in turn serve to
      // update Uout !
      int offset = elemOffset;
      uOut[ID] = Uin[offset];
      offset += arraySize;
      uOut[IP] = Uin[offset];
      offset += arraySize;
      uOut[IU] = Uin[offset];
      offset += arraySize;
      uOut[IV] = Uin[offset];
      offset += arraySize;
      uOut[IW] = Uin[offset];

      // update U with flux
      uOut[ID] += (flux[tx][ty][ID] - flux[tx + 1][ty][ID]) * dtdx;
      uOut[IP] += (flux[tx][ty][IP] - flux[tx + 1][ty][IP]) * dtdx;
      uOut[IU] += (flux[tx][ty][IU] - flux[tx + 1][ty][IU]) * dtdx;
      uOut[IV] += (flux[tx][ty][IV] - flux[tx + 1][ty][IV]) * dtdx;
      uOut[IW] += (flux[tx][ty][IW] - flux[tx + 1][ty][IW]) * dtdx;

    } // end update hydro along X
    __syncthreads();

    // solve Riemann problems at Y interfaces
    flux[tx][ty][ID] = ZERO_F;
    flux[tx][ty][IP] = ZERO_F;
    flux[tx][ty][IU] = ZERO_F;
    flux[tx][ty][IV] = ZERO_F;
    flux[tx][ty][IW] = ZERO_F;
    __syncthreads();
    if (i >= 2 and i < imax - 1 and tx > 1 and tx < UNSPLIT_BLOCK_DIMX_3D - 1 and j >= 2 and
        j < jmax - 1 and ty > 1 and ty < UNSPLIT_BLOCK_DIMY_3D - 1)
    {
      real_t qgdnv[NVAR_3D];

      // Solve Riemann problem at Y-interfaces and compute fluxes
      real_t(&qleft_y)[NVAR_3D] = qm2;
      real_t(&qright_y)[NVAR_3D] = qp0[1];
      // watchout swap IU and IV
      swap_value(qleft_y[IU], qleft_y[IV]);
      swap_value(qright_y[IU], qright_y[IV]);
      riemann<NVAR_3D>(qleft_y, qright_y, qgdnv, flux[tx][ty]);
    } //  end solve Riemann at Y interfaces
    __syncthreads();

    // update hydro along Y
    if (i >= 2 and i < imax - 1 and tx > 1 and tx < UNSPLIT_BLOCK_DIMX_3D - 2 and j >= 2 and
        j < jmax - 1 and ty > 1 and ty < UNSPLIT_BLOCK_DIMY_3D - 2)
    {
      // update U with flux : watchout! IU and IV are swapped !
      uOut[ID] += (flux[tx][ty][ID] - flux[tx][ty + 1][ID]) * dtdx;
      uOut[IP] += (flux[tx][ty][IP] - flux[tx][ty + 1][IP]) * dtdx;
      uOut[IU] += (flux[tx][ty][IV] - flux[tx][ty + 1][IV]) * dtdx;
      uOut[IV] += (flux[tx][ty][IU] - flux[tx][ty + 1][IU]) * dtdx;
      uOut[IW] += (flux[tx][ty][IW] - flux[tx][ty + 1][IW]) * dtdx;

    } // end update hydro along Y
    __syncthreads();

    // solve Riemann problems at Z interfaces
    real_t flux_z[NVAR_3D];
    if (i >= 2 and i < imax - 1 and tx > 1 and tx < UNSPLIT_BLOCK_DIMX_3D - 1 and j >= 2 and
        j < jmax - 1 and ty > 1 and ty < UNSPLIT_BLOCK_DIMY_3D - 1)
    {
      real_t qgdnv[NVAR_3D];

      // Solve Riemann problem at Z-interfaces and compute fluxes
      real_t(&qleft_z)[NVAR_3D] = qm3;
      real_t(&qright_z)[NVAR_3D] = qp0[2];
      // watchout swap IU and IW
      swap_value(qleft_z[IU], qleft_z[IW]);
      swap_value(qright_z[IU], qright_z[IW]);
      riemann<NVAR_3D>(qleft_z, qright_z, qgdnv, flux_z);
    } // end solve Riemann at Z interfaces
    __syncthreads();

    // update hydro along Z
    if (i >= 2 and i < imax - 1 and tx > 1 and tx < UNSPLIT_BLOCK_DIMX_3D - 2 and j >= 2 and
        j < jmax - 1 and ty > 1 and ty < UNSPLIT_BLOCK_DIMY_3D - 2)
    {
      /*
       * take care that update with flux_z is separated in two stages !!!
       */

      /*
       * update current position z.
       */
      // update U with flux : watchout! IU and IW are swapped !
      uOut[ID] += (flux_z[ID]) * dtdx;
      uOut[IP] += (flux_z[IP]) * dtdx;
      uOut[IU] += (flux_z[IW]) * dtdx;
      uOut[IV] += (flux_z[IV]) * dtdx;
      uOut[IW] += (flux_z[IU]) * dtdx;

      /*
       * update at position z-1.
       * Note that position z-1 has already been partially updated in
       * the previous iteration (for loop over k).
       */
      // update U with flux : watchout! IU and IW are swapped !
      int offset = elemOffset - pitch * jmax;
      Uout[offset] -= (flux_z[ID]) * dtdx;
      offset += arraySize;
      Uout[offset] -= (flux_z[IP]) * dtdx;
      offset += arraySize;
      Uout[offset] -= (flux_z[IW]) * dtdx;
      offset += arraySize;
      Uout[offset] -= (flux_z[IV]) * dtdx;
      offset += arraySize;
      Uout[offset] -= (flux_z[IU]) * dtdx;

      // actually perform the update on external device memory
      offset = elemOffset;
      Uout[offset] = uOut[ID];
      offset += arraySize;
      Uout[offset] = uOut[IP];
      offset += arraySize;
      Uout[offset] = uOut[IU];
      offset += arraySize;
      Uout[offset] = uOut[IV];
      offset += arraySize;
      Uout[offset] = uOut[IW];

    } // end update along Z
    __syncthreads();

    /*
     * swap planes
     */
    tmp = low;
    low = mid;
    mid = current;
    current = top;
    top = tmp;
    __syncthreads();

    /*
     * load new data (located in plane at z=k+2) and place them into top plane
     */
    if (k < kmax - 2)
    {
      if (i >= 0 and i < imax and j >= 0 and j < jmax)
      {

        // Gather conservative variables
        int offset = i + pitch * (j + jmax * (k + 2));
        uIn[ID] = Uin[offset];
        offset += arraySize;
        uIn[IP] = Uin[offset];
        offset += arraySize;
        uIn[IU] = Uin[offset];
        offset += arraySize;
        uIn[IV] = Uin[offset];
        offset += arraySize;
        uIn[IW] = Uin[offset];

        // Convert to primitive variables
        constoprim_3D(uIn, q[top][tx][ty], c[top]);
      }
    }
    __syncthreads();

  } // end for k

} //  kernel_godunov_unsplit_3d

#endif // KERNEL_GODUNOV_UNSPLIT_3D_CUH_
