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
 * \file kernel_godunov_unsplit_2d_v4.cuh
 * \brief Define version 4 of godunov kernel.
 */
#ifndef KERNEL_GODUNOV_UNSPLIT_2D_V4_CUH_
#define KERNEL_GODUNOV_UNSPLIT_2D_V4_CUH_

#include "real_type.h"
#include "constants.h"
#include "kernel_godunov_unsplit_2d_v4_utils.cuh"

/**
 * Unsplit Godunov kernel for 2D data (version 4).
 *
 * Primitive variables are computed here.
 *
 */
__global__ void
kernel_godunov_unsplit_2d_v4(const real_t * __restrict__ Uin,
                             real_t * uOut,
                             int      pitch,
                             int      imax,
                             int      jmax,
                             real_t   dtdx,
                             real_t   dtdy,
                             real_t   dt,
                             bool     gravityEnabled)
{
  // Block index
  const int bx = blockIdx.x;
  const int by = blockIdx.y;

  // Thread index
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;

  const int i = __mul24(bx, UNSPLIT_BLOCK_INNER_DIMX_2D_V4) + tx;
  const int j = __mul24(by, UNSPLIT_BLOCK_INNER_DIMY_2D_V4) + ty;

  const int arraySize = __umul24(pitch, jmax);
  const int elemOffset = __umul24(pitch, j) + i;

  // primitive variables inner + ghost cells
  __shared__ QShmem q;

  // reconstructed states on X interface
  __shared__ QReconsShmem qL;
  __shared__ QReconsShmem qR;

  // conservative variables
  real_t uIn[NVAR_2D];

  // riemann solver output
  real_t qgdnv[NVAR_2D];
  real_t flux_x[NVAR_2D];
  real_t flux_y[NVAR_2D];

  real_t * gravin = gParams.arrayList[A_GRAV];

  // compute primitive variables
  if (i >= 0 and i < imax and j >= 0 and j < jmax)
  {
    [[maybe_unused]] real_t c;

    // read conservative variables
    int offset = elemOffset;
    uIn[ID] = Uin[offset];
    offset += arraySize;
    uIn[IP] = Uin[offset];
    offset += arraySize;
    uIn[IU] = Uin[offset];
    offset += arraySize;
    uIn[IV] = Uin[offset];

    // Convert to primitive variables
    constoprim_2D(uIn, q, tx, ty, c);
  }
  __syncthreads();

  //
  // compute reconstructed on X faces
  //

  // clang-format off
  if (i > 0 and i < imax - 1 and tx > 0 and tx < UNSPLIT_BLOCK_DIMX_2D_V4 - 1 and
      j > 0 and j < jmax - 1 and ty > 0 and ty < UNSPLIT_BLOCK_DIMY_2D_V4 - 1)
  // clang-format on
  {

    // compute left and right state
    trace_unsplit_hydro_2d_by_direction_shmem(q, tx, ty, dtdx, dtdy, IX, qL, qR);

    if (gravityEnabled)
    {

      // we need to modify input to flux computation with
      // gravity predictor (half time step)
      qleft[IU] += HALF_F * dt * gravin[elemOffset + IX * arraySize];
      qleft[IV] += HALF_F * dt * gravin[elemOffset + IY * arraySize];

      qright[IU] += HALF_F * dt * gravin[elemOffset + IX * arraySize];
      qright[IV] += HALF_F * dt * gravin[elemOffset + IY * arraySize];

    } // end gravityEnabled
  }
  __syncthreads();

  // clang-format off
  if (i > 1 and i < imax - 1 and tx > 1 and tx < UNSPLIT_BLOCK_DIMX_2D_V4 - 1 and
      j > 1 and j < jmax - 1 and ty > 1 and ty < UNSPLIT_BLOCK_DIMY_2D_V4 - 1)
  // clang-format on
  {
    // Solve Riemann problem at X-interfaces and compute X-fluxes
    riemann<NVAR_2D>(qleft, qright, qgdnv, flux_x);

    deltaU[tx][ty][ID] += flux_x[ID] * dtdx;
    deltaU[tx][ty][IP] += flux_x[IP] * dtdx;
    deltaU[tx][ty][IU] += flux_x[IU] * dtdx;
    deltaU[tx][ty][IV] += flux_x[IV] * dtdx;
  }
  __syncthreads();

  // clang-format off
  if (i > 1 and i < imax - 1 and tx > 1 and tx < UNSPLIT_BLOCK_DIMX_2D_V4 - 1 and
      j > 1 and j < jmax - 1 and ty > 1 and ty < UNSPLIT_BLOCK_DIMY_2D_V4 - 1)
  // clang-format on
  {

    deltaU[tx - 1][ty][ID] -= flux_x[ID] * dtdx;
    deltaU[tx - 1][ty][IP] -= flux_x[IP] * dtdx;
    deltaU[tx - 1][ty][IU] -= flux_x[IU] * dtdx;
    deltaU[tx - 1][ty][IV] -= flux_x[IV] * dtdx;
  }
  __syncthreads();

  //
  // compute reconstructed on Y faces
  //

  // clang-format off
  if (i > 1 and i < imax - 1 and tx > 1 and tx < UNSPLIT_BLOCK_DIMX_2D_V4 - 1 and
      j > 1 and j < jmax - 1 and ty > 1 and ty < UNSPLIT_BLOCK_DIMY_2D_V4 - 1)
  // clang-format on
  {

    // load neighborhood
    for (int iVar = 0; iVar < NVAR_2D; iVar++)
    {
      qLoc[iVar] = q[tx][ty][iVar];
      qNeighbors[0][iVar] = q[tx + 1][ty][iVar];
      qNeighbors[1][iVar] = q[tx - 1][ty][iVar];
      qNeighbors[2][iVar] = q[tx][ty + 1][iVar];
      qNeighbors[3][iVar] = q[tx][ty - 1][iVar];
    }

    // compute slopes in current cell
    slope_unsplit_hydro_2d(qLoc, qNeighbors[0], qNeighbors[1], qNeighbors[2], qNeighbors[3], dq);

    // left interface : right state
    trace_unsplit_hydro_2d_by_direction(qLoc, dq, dtdx, dtdy, FACE_YMIN, qright);

    // get primitive variables state vector in left neighbor along Y
    for (int iVar = 0; iVar < NVAR_2D; iVar++)
    {
      qLoc[iVar] = q[tx][ty - 1][iVar];
      qNeighbors[0][iVar] = q[tx + 1][ty - 1][iVar];
      qNeighbors[1][iVar] = q[tx - 1][ty - 1][iVar];
      qNeighbors[2][iVar] = q[tx][ty][iVar];
      qNeighbors[3][iVar] = q[tx][ty - 2][iVar];
    }

    // compute slopes in left neighbor along X
    slope_unsplit_hydro_2d(qLoc, qNeighbors[0], qNeighbors[1], qNeighbors[2], qNeighbors[3], dq);

    // left interface : left state
    trace_unsplit_hydro_2d_by_direction(qLoc, dq, dtdx, dtdy, FACE_YMAX, qleft);

    if (gravityEnabled)
    {

      // we need to modify input to flux computation with
      // gravity predictor (half time step)
      qleft[IU] += HALF_F * dt * gravin[elemOffset + IX * arraySize];
      qleft[IV] += HALF_F * dt * gravin[elemOffset + IY * arraySize];

      qright[IU] += HALF_F * dt * gravin[elemOffset + IX * arraySize];
      qright[IV] += HALF_F * dt * gravin[elemOffset + IY * arraySize];

    } // end gravityEnabled

    // Solve Riemann problem at X-interfaces and compute Y-fluxes
    swap_value(qleft[IU], qleft[IV]);
    swap_value(qright[IU], qright[IV]);
    riemann<NVAR_2D>(qleft, qright, qgdnv, flux_y);

    // watchout IU / IV are swapped
    deltaU[tx][ty][ID] += flux_y[ID] * dtdy;
    deltaU[tx][ty][IP] += flux_y[IP] * dtdy;
    deltaU[tx][ty][IU] += flux_y[IV] * dtdy;
    deltaU[tx][ty][IV] += flux_y[IU] * dtdy;
  }
  __syncthreads();

  // clang-format off
  if (i > 1 and i < imax - 1 and tx > 1 and tx < UNSPLIT_BLOCK_DIMX_2D_V4 - 1 and
      j > 1 and j < jmax - 1 and ty > 1 and ty < UNSPLIT_BLOCK_DIMY_2D_V4 - 1)
  // clang-format on
  {

    // watchout IU / IV are swapped
    deltaU[tx][ty - 1][ID] -= flux_y[ID] * dtdy;
    deltaU[tx][ty - 1][IP] -= flux_y[IP] * dtdy;
    deltaU[tx][ty - 1][IU] -= flux_y[IV] * dtdy;
    deltaU[tx][ty - 1][IV] -= flux_y[IU] * dtdy;
  }
  __syncthreads();

  // clang-format off
  if (i > 1 and i < imax - 2 and tx > 1 and tx < UNSPLIT_BLOCK_DIMX_2D_V4 - 2 and
      j > 1 and j < jmax - 2 and ty > 1 and ty < UNSPLIT_BLOCK_DIMY_2D_V4 - 2)
  // clang-format on
  {

    // actually perform update on external device memory
    int offset = elemOffset;
    Uout[offset] += deltaU[tx][ty][ID];
    offset += arraySize;
    Uout[offset] += deltaU[tx][ty][IP];
    offset += arraySize;
    Uout[offset] += deltaU[tx][ty][IU];
    offset += arraySize;
    Uout[offset] += deltaU[tx][ty][IV];
  }

} // kernel_godunov_unsplit_2d_v4

#endif // KERNEL_GODUNOV_UNSPLIT_2D_V4_CUH_
