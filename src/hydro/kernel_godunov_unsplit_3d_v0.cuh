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
 * \file kernel_godunov_unsplit_3d_v0.cuh
 * \brief Define 3d godunov kernel (version 0).
 */
#ifndef KERNEL_GODUNOV_UNSPLIT_3D_V0_CUH_
#define KERNEL_GODUNOV_UNSPLIT_3D_V0_CUH_

#include "real_type.h"
#include "constants.h"

// 3D-kernel block dimensions
#ifdef USE_DOUBLE
// newer tested
#  define UNSPLIT_BLOCK_DIMX_3D_V0 16
#  define UNSPLIT_BLOCK_INNER_DIMX_3D_V0 (UNSPLIT_BLOCK_DIMX_3D_V0 - 4)
#  define UNSPLIT_BLOCK_DIMY_3D_V0 12
#  define UNSPLIT_BLOCK_INNER_DIMY_3D_V0 (UNSPLIT_BLOCK_DIMY_3D_V0 - 4)
#else // simple precision
#  define UNSPLIT_BLOCK_DIMX_3D_V0 16
#  define UNSPLIT_BLOCK_INNER_DIMX_3D_V0 (UNSPLIT_BLOCK_DIMX_3D_V0 - 4)
#  define UNSPLIT_BLOCK_DIMY_3D_V0 24
#  define UNSPLIT_BLOCK_INNER_DIMY_3D_V0 (UNSPLIT_BLOCK_DIMY_3D_V0 - 4)
#endif // USE_DOUBLE

/**
 * Unsplit Godunov kernel for 3D data version 0.
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
kernel_godunov_unsplit_3d_v0(const real_t * __restrict__ qData,
                             real_t * Uout,
                             int      pitch,
                             int      imax,
                             int      jmax,
                             int      kmax,
                             real_t   dtdx,
                             real_t   dtdy,
                             real_t   dtdz,
                             real_t   dt,
                             bool     gravityEnabled)
{

  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;

  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;

  const int i = __mul24(bx, UNSPLIT_BLOCK_INNER_DIMX_3D_V0) + tx;
  const int j = __mul24(by, UNSPLIT_BLOCK_INNER_DIMY_3D_V0) + ty;

  // 3D array size
  const int arraySize = pitch * jmax * kmax;

  // we always store in shared 4 consecutive XY-plans
  __shared__ real_t q[4][UNSPLIT_BLOCK_DIMX_3D_V0][UNSPLIT_BLOCK_DIMY_3D_V0][NVAR_3D];

  real_t * gravin = gParams.arrayList[A_GRAV];

  // index to address the 4 plans of data
  int low, mid, current, top, tmp;
  low = 0;
  mid = 1;
  current = 2;
  top = 3;

  // primitive variables
  // real_t qLoc[NVAR_3D];

  // slopes
  real_t dq[THREE_D][NVAR_3D];
  real_t dqN[THREE_D][NVAR_3D];

  // qNeighbors is used for trace computations
  // real_t qNeighbors[2*THREE_D][NVAR_3D];

  // reconstructed state on cell faces
  // aka riemann solver input
  real_t qleft[NVAR_3D];
  real_t qright[NVAR_3D];

  real_t qgdnv[NVAR_3D];
  real_t flux_x[NVAR_3D];
  real_t flux_y[NVAR_3D];
  real_t flux_z[NVAR_3D];


  /*
   * initialize q with the first 4 plans
   */
  for (int k = 0, elemOffset = i + pitch * j; k < 4; ++k, elemOffset += (pitch * jmax))
  {

    if (i >= 0 and i < imax and j >= 0 and j < jmax)
    {

      // Gather conservative variables
      int offset = elemOffset;
      q[k][tx][ty][ID] = qData[offset];
      offset += arraySize;
      q[k][tx][ty][IP] = qData[offset];
      offset += arraySize;
      q[k][tx][ty][IU] = qData[offset];
      offset += arraySize;
      q[k][tx][ty][IV] = qData[offset];
      offset += arraySize;
      q[k][tx][ty][IW] = qData[offset];
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

    if (i > 1 and i < imax - 1 and tx > 1 and tx < UNSPLIT_BLOCK_DIMX_3D_V0 - 1 and j > 1 and
        j < jmax - 1 and ty > 1 and ty < UNSPLIT_BLOCK_DIMY_3D_V0 - 1)
    {

      //
      // along X
      //

      // compute slopes in current cell
      slope_unsplit_3d(q[current][tx][ty],
                       q[current][tx + 1][ty],
                       q[current][tx - 1][ty],
                       q[current][tx][ty + 1],
                       q[current][tx][ty - 1],
                       q[top][tx][ty],
                       q[mid][tx][ty],
                       dq);

      // left interface : right state
      trace_unsplit_hydro_3d_by_direction(
        q[current][tx][ty], dq, dtdx, dtdy, dtdz, FACE_XMIN, qright);

      // compute slopes in neighbor x - 1
      slope_unsplit_3d(q[current][tx - 1][ty],
                       q[current][tx][ty],
                       q[current][tx - 2][ty],
                       q[current][tx - 1][ty + 1],
                       q[current][tx - 1][ty - 1],
                       q[top][tx - 1][ty],
                       q[mid][tx - 1][ty],
                       dqN);

      // left interface : left state
      trace_unsplit_hydro_3d_by_direction(
        q[current][tx - 1][ty], dqN, dtdx, dtdy, dtdz, FACE_XMAX, qleft);

      // gravity predictor on velocity components of qp0's
      if (gravityEnabled)
      {
        qleft[IU] += HALF_F * dt * gravin[elemOffset + IX * arraySize];
        qleft[IV] += HALF_F * dt * gravin[elemOffset + IY * arraySize];
        qleft[IW] += HALF_F * dt * gravin[elemOffset + IZ * arraySize];

        qright[IU] += HALF_F * dt * gravin[elemOffset + IX * arraySize];
        qright[IV] += HALF_F * dt * gravin[elemOffset + IY * arraySize];
        qright[IW] += HALF_F * dt * gravin[elemOffset + IZ * arraySize];
      }

      // Solve Riemann problem at X-interfaces and compute X-fluxes
      riemann<NVAR_3D>(qleft, qright, qgdnv, flux_x);

      //
      // along Y
      //

      // left interface : right state
      trace_unsplit_hydro_3d_by_direction(
        q[current][tx][ty], dq, dtdx, dtdy, dtdz, FACE_YMIN, qright);

      // compute slopes in neighbor y - 1
      slope_unsplit_3d(q[current][tx][ty - 1],
                       q[current][tx + 1][ty - 1],
                       q[current][tx - 1][ty - 1],
                       q[current][tx][ty],
                       q[current][tx][ty - 2],
                       q[top][tx][ty - 1],
                       q[mid][tx][ty - 1],
                       dqN);

      // left interface : left state
      trace_unsplit_hydro_3d_by_direction(
        q[current][tx][ty - 1], dqN, dtdx, dtdy, dtdz, FACE_YMAX, qleft);

      // gravity predictor on velocity components
      if (gravityEnabled)
      {
        qleft[IU] += HALF_F * dt * gravin[elemOffset + IX * arraySize];
        qleft[IV] += HALF_F * dt * gravin[elemOffset + IY * arraySize];
        qleft[IW] += HALF_F * dt * gravin[elemOffset + IZ * arraySize];

        qright[IU] += HALF_F * dt * gravin[elemOffset + IX * arraySize];
        qright[IV] += HALF_F * dt * gravin[elemOffset + IY * arraySize];
        qright[IW] += HALF_F * dt * gravin[elemOffset + IZ * arraySize];
      }

      // Solve Riemann problem at Y-interfaces and compute Y-fluxes
      swap_value(qleft[IU], qleft[IV]);
      swap_value(qright[IU], qright[IV]);
      riemann<NVAR_3D>(qleft, qright, qgdnv, flux_y);

      //
      // along Z
      //

      // left interface : right state
      trace_unsplit_hydro_3d_by_direction(
        q[current][tx][ty], dq, dtdx, dtdy, dtdz, FACE_ZMIN, qright);

      // compute slopes in neighbor z - 1
      slope_unsplit_3d(q[mid][tx][ty],
                       q[mid][tx + 1][ty],
                       q[mid][tx - 1][ty],
                       q[mid][tx][ty + 1],
                       q[mid][tx][ty - 1],
                       q[current][tx][ty],
                       q[low][tx][ty],
                       dqN);

      // left interface : left state
      trace_unsplit_hydro_3d_by_direction(q[mid][tx][ty], dqN, dtdx, dtdy, dtdz, FACE_ZMAX, qleft);

      // gravity predictor on velocity components
      if (gravityEnabled)
      {
        qleft[IU] += HALF_F * dt * gravin[elemOffset + IX * arraySize];
        qleft[IV] += HALF_F * dt * gravin[elemOffset + IY * arraySize];
        qleft[IW] += HALF_F * dt * gravin[elemOffset + IZ * arraySize];

        qright[IU] += HALF_F * dt * gravin[elemOffset + IX * arraySize];
        qright[IV] += HALF_F * dt * gravin[elemOffset + IY * arraySize];
        qright[IW] += HALF_F * dt * gravin[elemOffset + IZ * arraySize];
      }

      // Solve Riemann problem at Y-interfaces and compute Y-fluxes
      swap_value(qleft[IU], qleft[IW]);
      swap_value(qright[IU], qright[IW]);
      riemann<NVAR_3D>(qleft, qright, qgdnv, flux_z);
    }
    // end trace/slope computations, we have all we need to
    // compute qleft/qright now
    __syncthreads();

    /*
     * Now, perform update
     */
    // a nice trick: reuse q[low] to store deltaU, as we don't need then anymore
    // also remember that q[low] will become q[top] after the flux
    // computations and hydro update
    real_t(&deltaU)[UNSPLIT_BLOCK_DIMX_3D_V0][UNSPLIT_BLOCK_DIMY_3D_V0][NVAR_3D] = q[low];

    //
    deltaU[tx][ty][ID] = ZERO_F;
    deltaU[tx][ty][IP] = ZERO_F;
    deltaU[tx][ty][IU] = ZERO_F;
    deltaU[tx][ty][IV] = ZERO_F;
    deltaU[tx][ty][IW] = ZERO_F;
    //__syncthreads();

    if (i >= 2 and i < imax - 1 and tx > 1 and tx < UNSPLIT_BLOCK_DIMX_3D_V0 - 1 and j >= 2 and
        j < jmax - 1 and ty > 1 and ty < UNSPLIT_BLOCK_DIMY_3D_V0 - 1)
    {
      // take for flux_y : IU and IV are swapped
      // take for flux_z : IU and IW are swapped
      deltaU[tx][ty][ID] += (flux_x[ID] * dtdx + flux_y[ID] * dtdy + flux_z[ID] * dtdz);
      deltaU[tx][ty][IP] += (flux_x[IP] * dtdx + flux_y[IP] * dtdy + flux_z[IP] * dtdz);
      deltaU[tx][ty][IU] += (flux_x[IU] * dtdx + flux_y[IV] * dtdy + flux_z[IW] * dtdz);
      deltaU[tx][ty][IV] += (flux_x[IV] * dtdx + flux_y[IU] * dtdy + flux_z[IV] * dtdz);
      deltaU[tx][ty][IW] += (flux_x[IW] * dtdx + flux_y[IW] * dtdy + flux_z[IU] * dtdz);
    }
    __syncthreads();

    if (i >= 2 and i < imax - 1 and tx > 1 and tx < UNSPLIT_BLOCK_DIMX_3D_V0 - 1 and j >= 2 and
        j < jmax - 1 and ty > 1 and ty < UNSPLIT_BLOCK_DIMY_3D_V0 - 1)
    {

      deltaU[tx - 1][ty][ID] -= flux_x[ID] * dtdx;
      deltaU[tx - 1][ty][IP] -= flux_x[IP] * dtdx;
      deltaU[tx - 1][ty][IU] -= flux_x[IU] * dtdx;
      deltaU[tx - 1][ty][IV] -= flux_x[IV] * dtdx;
      deltaU[tx - 1][ty][IW] -= flux_x[IW] * dtdx;
    }
    __syncthreads();

    if (i >= 2 and i < imax - 1 and tx > 1 and tx < UNSPLIT_BLOCK_DIMX_3D_V0 - 1 and j >= 2 and
        j < jmax - 1 and ty > 1 and ty < UNSPLIT_BLOCK_DIMY_3D_V0 - 1)
    {

      // take for flux_y : IU and IV are swapped
      deltaU[tx][ty - 1][ID] -= flux_y[ID] * dtdy;
      deltaU[tx][ty - 1][IP] -= flux_y[IP] * dtdy;
      deltaU[tx][ty - 1][IU] -= flux_y[IV] * dtdy;
      deltaU[tx][ty - 1][IV] -= flux_y[IU] * dtdy;
      deltaU[tx][ty - 1][IW] -= flux_y[IW] * dtdy;
    }
    __syncthreads();


    // update at z and z-1
    if (i >= 2 and i < imax - 1 and tx > 1 and tx < UNSPLIT_BLOCK_DIMX_3D_V0 - 2 and j >= 2 and
        j < jmax - 1 and ty > 1 and ty < UNSPLIT_BLOCK_DIMY_3D_V0 - 2)
    {

      /*
       * update at position z-1.
       * Note that position z-1 has already been partially updated in
       * the previous iteration (for loop over k).
       */
      // update U with flux : watchout! IU and IW are swapped !
      int offset = elemOffset - pitch * jmax;
      Uout[offset] -= (flux_z[ID]) * dtdz;
      offset += arraySize;
      Uout[offset] -= (flux_z[IP]) * dtdz;
      offset += arraySize;
      Uout[offset] -= (flux_z[IW]) * dtdz;
      offset += arraySize;
      Uout[offset] -= (flux_z[IV]) * dtdz;
      offset += arraySize;
      Uout[offset] -= (flux_z[IU]) * dtdz;

      // actually perform the update on external device memory
      offset = elemOffset;
      Uout[offset] += deltaU[tx][ty][ID];
      offset += arraySize;
      Uout[offset] += deltaU[tx][ty][IP];
      offset += arraySize;
      Uout[offset] += deltaU[tx][ty][IU];
      offset += arraySize;
      Uout[offset] += deltaU[tx][ty][IV];
      offset += arraySize;
      Uout[offset] += deltaU[tx][ty][IW];

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

        // Gather primitive variables
        int offset = i + pitch * (j + jmax * (k + 2));
        q[top][tx][ty][ID] = qData[offset];
        offset += arraySize;
        q[top][tx][ty][IP] = qData[offset];
        offset += arraySize;
        q[top][tx][ty][IU] = qData[offset];
        offset += arraySize;
        q[top][tx][ty][IV] = qData[offset];
        offset += arraySize;
        q[top][tx][ty][IW] = qData[offset];
      }
    }
    __syncthreads();

  } // end for k

} //  kernel_godunov_unsplit_3d_v0

#endif // KERNEL_GODUNOV_UNSPLIT_3D_V0_CUH_
