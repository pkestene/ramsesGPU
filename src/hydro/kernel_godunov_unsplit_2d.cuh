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
 * \file kernel_godunov_unsplit_2d.cuh
 * \brief Define 2d godunov kernel.
 */
#ifndef KERNEL_GODUNOV_UNSPLIT_2D_CUH_
#define KERNEL_GODUNOV_UNSPLIT_2D_CUH_

#include "real_type.h"
#include "constants.h"

// 2D-kernel block dimensions
#ifdef USE_DOUBLE
#  define UNSPLIT_BLOCK_DIMX_2D 16
#  define UNSPLIT_BLOCK_INNER_DIMX_2D (UNSPLIT_BLOCK_DIMX_2D - 4)
#  define UNSPLIT_BLOCK_DIMY_2D 12
#  define UNSPLIT_BLOCK_INNER_DIMY_2D (UNSPLIT_BLOCK_DIMY_2D - 4)
#else // simple precision
#  define UNSPLIT_BLOCK_DIMX_2D 16
#  define UNSPLIT_BLOCK_INNER_DIMX_2D (UNSPLIT_BLOCK_DIMX_2D - 4)
#  define UNSPLIT_BLOCK_DIMY_2D 24
#  define UNSPLIT_BLOCK_INNER_DIMY_2D (UNSPLIT_BLOCK_DIMY_2D - 4)
#endif // USE_DOUBLE

/**
 * Unsplit Godunov kernel for 2D data.
 *
 * This kernel doesn't eat so much shared memory, but to the price of
 * more computations...
 * We remove array qm_x1 and qm_y2, and recompute what is needed by
 * each thread.
 */
__global__ void
kernel_godunov_unsplit_2d(const real_t * __restrict__ Uin,
                          real_t * Uout,
                          int      pitch,
                          int      imax,
                          int      jmax,
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

  const int i = __mul24(bx, UNSPLIT_BLOCK_INNER_DIMX_2D) + tx;
  const int j = __mul24(by, UNSPLIT_BLOCK_INNER_DIMY_2D) + ty;

  const int arraySize = __umul24(pitch, jmax);
  const int elemOffset = __umul24(pitch, j) + i;

  __shared__ real_t q[UNSPLIT_BLOCK_DIMX_2D][UNSPLIT_BLOCK_DIMY_2D][NVAR_2D];
  __shared__ real_t flux_y[UNSPLIT_BLOCK_DIMX_2D][UNSPLIT_BLOCK_DIMY_2D][NVAR_2D];
  real_t            qm[TWO_D][NVAR_2D];
  real_t            qm1[NVAR_2D];
  real_t            qm2[NVAR_2D];

  real_t qp[TWO_D][NVAR_2D];
  real_t qp0[TWO_D][NVAR_2D];

  // conservative variables
  real_t uIn[NVAR_2D];
  real_t uOut[NVAR_2D];
  real_t c;

  real_t * gravin = gParams.arrayList[A_GRAV];

  // load U and convert to primitive variables
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

    // copy input state into uOut that will become output state
    uOut[ID] = uIn[ID];
    uOut[IP] = uIn[IP];
    uOut[IU] = uIn[IU];
    uOut[IV] = uIn[IV];

    // Convert to primitive variables
    constoprim_2D(uIn, q[tx][ty], c);
  }
  __syncthreads();

  // clang-format off
  if (i > 1 and i < imax - 1 and tx > 1 and tx < UNSPLIT_BLOCK_DIMX_2D - 1 and
      j > 1 and j < jmax - 1 and ty > 1 and ty < UNSPLIT_BLOCK_DIMY_2D - 1)
  // clang-format on
  {
    real_t qNeighbors[2 * TWO_D][NVAR_2D];

    // Characteristic tracing : compute qp0
    for (int iVar = 0; iVar < NVAR_2D; iVar++)
    {
      qNeighbors[0][iVar] = q[tx + 1][ty][iVar];
      qNeighbors[1][iVar] = q[tx - 1][ty][iVar];
      qNeighbors[2][iVar] = q[tx][ty + 1][iVar];
      qNeighbors[3][iVar] = q[tx][ty - 1][iVar];
    }
    trace_unsplit<TWO_D, NVAR_2D>(q[tx][ty], qNeighbors, c, dtdx, qm, qp0);

    // gravity predictor on velocity component of qp0's
    if (gravityEnabled)
    {
      qp0[0][IU] += HALF_F * dt * gravin[elemOffset + IX * arraySize];
      qp0[0][IV] += HALF_F * dt * gravin[elemOffset + IY * arraySize];

      qp0[1][IU] += HALF_F * dt * gravin[elemOffset + IX * arraySize];
      qp0[1][IV] += HALF_F * dt * gravin[elemOffset + IY * arraySize];
    }

    // Characteristic tracing : compute qm_x[1]
    for (int iVar = 0; iVar < NVAR_2D; iVar++)
    {
      qNeighbors[0][iVar] = q[tx][ty][iVar];
      qNeighbors[1][iVar] = q[tx - 2][ty][iVar];
      qNeighbors[2][iVar] = q[tx - 1][ty + 1][iVar];
      qNeighbors[3][iVar] = q[tx - 1][ty - 1][iVar];
    }
    trace_unsplit<TWO_D, NVAR_2D>(q[tx - 1][ty], qNeighbors, c, dtdx, qm, qp);
    for (int iVar = 0; iVar < NVAR_2D; iVar++)
    {
      qm1[iVar] = qm[0][iVar];
    }

    // gravity predictor on velocity component of qm1
    if (gravityEnabled)
    {
      qm1[IU] += HALF_F * dt * gravin[elemOffset - 1 + IX * arraySize];
      qm1[IV] += HALF_F * dt * gravin[elemOffset - 1 + IY * arraySize];
    }

    // Characteristic tracing : compute qm_y[2]
    for (int iVar = 0; iVar < NVAR_2D; iVar++)
    {
      qNeighbors[0][iVar] = q[tx + 1][ty - 1][iVar];
      qNeighbors[1][iVar] = q[tx - 1][ty - 1][iVar];
      qNeighbors[2][iVar] = q[tx][ty][iVar];
      qNeighbors[3][iVar] = q[tx][ty - 2][iVar];
    }
    trace_unsplit<TWO_D, NVAR_2D>(q[tx][ty - 1], qNeighbors, c, dtdx, qm, qp);
    for (int iVar = 0; iVar < NVAR_2D; iVar++)
    {
      qm2[iVar] = qm[1][iVar];
    }

    // gravity predictor on velocity component of qm2
    if (gravityEnabled)
    {
      qm2[IU] += HALF_F * dt * gravin[elemOffset - pitch + IX * arraySize];
      qm2[IV] += HALF_F * dt * gravin[elemOffset - pitch + IY * arraySize];
    }
  }
  __syncthreads();

  // re-use q as flux_x
  real_t(&flux_x)[UNSPLIT_BLOCK_DIMX_2D][UNSPLIT_BLOCK_DIMY_2D][NVAR_2D] = q;
  flux_x[tx][ty][ID] = ZERO_F;
  flux_x[tx][ty][IP] = ZERO_F;
  flux_x[tx][ty][IU] = ZERO_F;
  flux_x[tx][ty][IV] = ZERO_F;
  flux_y[tx][ty][ID] = ZERO_F;
  flux_y[tx][ty][IP] = ZERO_F;
  flux_y[tx][ty][IU] = ZERO_F;
  flux_y[tx][ty][IV] = ZERO_F;
  __syncthreads();

  // clang-format off
  if (i >= 2 and i < imax - 1 and tx > 1 and tx < UNSPLIT_BLOCK_DIMX_2D - 1 and
      j >= 2 and j < jmax - 1 and ty > 1 and ty < UNSPLIT_BLOCK_DIMY_2D - 1)
  // clang-format on
  {
    real_t qgdnv[NVAR_2D];

    // Solve Riemann problem at X-interfaces and compute fluxes
    real_t(&qleft_x)[NVAR_2D] = qm1;
    real_t(&qright_x)[NVAR_2D] = qp0[0];
    riemann<NVAR_2D>(qleft_x, qright_x, qgdnv, flux_x[tx][ty]);

    // Solve Riemann problem at Y-interfaces and compute fluxes
    real_t(&qleft_y)[NVAR_2D] = qm2;
    real_t(&qright_y)[NVAR_2D] = qp0[1];
    // watchout swap IU and IV
    swap_value(qleft_y[IU], qleft_y[IV]);
    swap_value(qright_y[IU], qright_y[IV]);
    riemann<NVAR_2D>(qleft_y, qright_y, qgdnv, flux_y[tx][ty]);
  }
  __syncthreads();

  // clang-format off
  if (i >= 2 and i < imax - 1 and tx > 1 and tx < UNSPLIT_BLOCK_DIMX_2D - 2 and
      j >= 2 and j < jmax - 1 and ty > 1 and ty < UNSPLIT_BLOCK_DIMY_2D - 2)
  // clang-format on
  {
    // update U with flux_x
    uOut[ID] += (flux_x[tx][ty][ID] - flux_x[tx + 1][ty][ID]) * dtdx;
    uOut[IP] += (flux_x[tx][ty][IP] - flux_x[tx + 1][ty][IP]) * dtdx;
    uOut[IU] += (flux_x[tx][ty][IU] - flux_x[tx + 1][ty][IU]) * dtdx;
    uOut[IV] += (flux_x[tx][ty][IV] - flux_x[tx + 1][ty][IV]) * dtdx;

    // update U with flux_y : watchout! IU and IV are swapped !
    uOut[ID] += (flux_y[tx][ty][ID] - flux_y[tx][ty + 1][ID]) * dtdx;
    uOut[IP] += (flux_y[tx][ty][IP] - flux_y[tx][ty + 1][IP]) * dtdx;
    uOut[IU] += (flux_y[tx][ty][IV] - flux_y[tx][ty + 1][IV]) * dtdx;
    uOut[IV] += (flux_y[tx][ty][IU] - flux_y[tx][ty + 1][IU]) * dtdx;

    // update momentum using gravity source term
    if (gravityEnabled)
    {
      uOut[IU] += HALF_F * (uIn[ID] + uOut[ID]) * dt * gravin[elemOffset + IX * arraySize] * dt;
      uOut[IV] += HALF_F * (uIn[ID] + uOut[ID]) * dt * gravin[elemOffset + IY * arraySize] * dt;
    }

    // actually perform update on external device memory
    int offset = elemOffset;
    Uout[offset] = uOut[ID];
    offset += arraySize;
    Uout[offset] = uOut[IP];
    offset += arraySize;
    Uout[offset] = uOut[IU];
    offset += arraySize;
    Uout[offset] = uOut[IV];
  }

} // kernel_godunov_unsplit_2d

#endif // KERNEL_GODUNOV_UNSPLIT_2D_CUH_
