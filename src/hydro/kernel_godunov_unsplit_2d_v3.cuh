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
 * \file kernel_godunov_unsplit_2d_v3.cuh
 * \brief Define version 4 of godunov kernel.
 */
#ifndef KERNEL_GODUNOV_UNSPLIT_2D_V3_CUH_
#define KERNEL_GODUNOV_UNSPLIT_2D_V3_CUH_

#include "real_type.h"
#include "constants.h"
#include "kernel_godunov_unsplit_2d_v3_utils.cuh"

/**
 * Unsplit Godunov kernel for 2D data (version 4).
 *
 * Primitive variables are computed here.
 *
 */
__global__ void
kernel_godunov_unsplit_2d_v3(const real_t * __restrict__ Uin,
                             real_t * __restrict__ Uout,
                             int    pitch,
                             int    imax,
                             int    jmax,
                             real_t dtdx,
                             real_t dtdy,
                             real_t dt,
                             bool   gravityEnabled)
{
  // Block index
  auto const & bx = blockIdx.x;
  auto const & by = blockIdx.y;

  // block dimension
  auto const dim_x = blockDim.x;
  auto const dim_y = blockDim.y;

  // Thread index
  auto const & tx = threadIdx.x;
  auto const & ty = threadIdx.y;

  const int i = __mul24(bx, dim_x - 4) + tx;
  const int j = __mul24(by, dim_y - 4) + ty;

  const int arraySize = __umul24(pitch, jmax);
  const int elemOffset = __umul24(pitch, j) + i;

  extern __shared__ char storage[];

  void * qprim_data_ptr = reinterpret_cast<void *>(&storage);
  void * qm_data_ptr =
    reinterpret_cast<void *>(&storage[QPrim::required_size_in_bytes(dim_x, dim_y)]);
  void * qp_data_ptr =
    reinterpret_cast<void *>(&storage[QPrim::required_size_in_bytes(dim_x, dim_y) +
                                      QRecons::required_size_in_bytes(dim_x, dim_y)]);
  void * flux_data_ptr =
    reinterpret_cast<void *>(&storage[QPrim::required_size_in_bytes(dim_x, dim_y) +
                                      2 * QRecons::required_size_in_bytes(dim_x, dim_y)]);

  // static allocation of all containers (this will call default constructor that does nothing)
  __shared__ QPrim   q;
  __shared__ QRecons qm;
  __shared__ QRecons qp;
  __shared__ QRecons flux;

  // call constructor at designated address with placement new (only 1 thread in block)
  if (tx == 0 and ty == 0)
  {
    ::new (&q) QPrim(qprim_data_ptr, dim_x, dim_y);
    ::new (&qm) QRecons(qm_data_ptr, dim_x, dim_y);
    ::new (&qp) QRecons(qp_data_ptr, dim_x, dim_y);
    ::new (&flux) QRecons(flux_data_ptr, dim_x, dim_y);
  }
  __syncthreads();

  // conservative variables
  real_t uIn[NVAR_2D];

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
  if (i > 0 and i < imax - 1 and tx > 0 and tx < dim_x - 1 and
      j > 0 and j < jmax - 1 and ty > 1 and ty < dim_y - 2)
  // clang-format on
  {

    // compute left and right state
    trace_unsplit_hydro_2d_by_direction_shmem(q, tx, ty, dtdx, dtdy, IX, qm, qp);

    if (gravityEnabled)
    {
      const int tx2 = tx - 1;
      const int ty2 = ty - 2;

      // we need to modify input to flux computation with
      // gravity predictor (half time step)
      qm(tx2, ty2, IU, IX) += HALF_F * dt * gravin[elemOffset + IX * arraySize];
      qm(tx2, ty2, IV, IX) += HALF_F * dt * gravin[elemOffset + IY * arraySize];

      qp(tx2, ty2, IU, IX) += HALF_F * dt * gravin[elemOffset + IX * arraySize];
      qp(tx2, ty2, IV, IX) += HALF_F * dt * gravin[elemOffset + IY * arraySize];

    } // end gravityEnabled
  }
  __syncthreads();

  //
  // Compute fluxes along X
  //

  // clang-format off
  if (i > 1 and i < imax - 1 and tx > 1 and tx < dim_x - 1 and
      j > 1 and j < jmax - 1 and ty > 1 and ty < dim_y - 2)
  // clang-format on
  {
    const int tx2 = tx - 1;
    const int ty2 = ty - 2;

    // Solve Riemann problem at X-interfaces and compute X-fluxes, and output the fluxes
    riemann<2, IX>(qm, qp, tx2, ty2, flux);
  }
  __syncthreads();

  //
  // Update along X
  //

  // clang-format off
  if (i > 1 and i < imax - 2 and tx > 1 and tx < dim_x - 2 and
      j > 1 and j < jmax - 2 and ty > 1 and ty < dim_y - 2)
  // clang-format on
  {
    const int tx2 = tx - 1;
    const int ty2 = ty - 2;

    int offset = elemOffset;
    Uout[offset] += (flux(tx2, ty2, ID, IX) - flux(tx2 + 1, ty2, ID, IX)) * dtdx;
    offset += arraySize;
    Uout[offset] += (flux(tx2, ty2, IP, IX) - flux(tx2 + 1, ty2, IP, IX)) * dtdx;
    offset += arraySize;
    Uout[offset] += (flux(tx2, ty2, IU, IX) - flux(tx2 + 1, ty2, IU, IX)) * dtdx;
    offset += arraySize;
    Uout[offset] += (flux(tx2, ty2, IV, IX) - flux(tx2 + 1, ty2, IV, IX)) * dtdx;
  }
  __syncthreads();

  //
  // compute reconstructed on Y faces
  //

  // clang-format off
  if (i > 0 and i < imax - 1 and tx > 1 and tx < dim_x - 2 and
      j > 0 and j < jmax - 1 and ty > 0 and ty < dim_y - 1)
  // clang-format on
  {

    // compute left and right state
    trace_unsplit_hydro_2d_by_direction_shmem(q, tx, ty, dtdx, dtdy, IY, qm, qp);

    if (gravityEnabled)
    {
      const int tx2 = tx - 2;
      const int ty2 = ty - 1;

      // we need to modify input to flux computation with
      // gravity predictor (half time step)
      qm(tx2, ty2, IU, IY) += HALF_F * dt * gravin[elemOffset + IX * arraySize];
      qm(tx2, ty2, IV, IY) += HALF_F * dt * gravin[elemOffset + IY * arraySize];

      qp(tx2, ty2, IU, IY) += HALF_F * dt * gravin[elemOffset + IX * arraySize];
      qp(tx2, ty2, IV, IY) += HALF_F * dt * gravin[elemOffset + IY * arraySize];

    } // end gravityEnabled
  }
  __syncthreads();

  //
  // Compute fluxes along Y
  //

  // clang-format off
  if (i > 1 and i < imax - 1 and tx > 1 and tx < dim_x - 2 and
      j > 1 and j < jmax - 1 and ty > 1 and ty < dim_y - 1)
  // clang-format on
  {
    const int tx2 = tx - 2;
    const int ty2 = ty - 1;

    // Solve Riemann problem at X-interfaces and compute X-fluxes, and output the fluxes
    riemann<2, IY>(qm, qp, tx2, ty2, flux);
  }
  __syncthreads();

  //
  // Update along Y
  //

  // clang-format off
  if (i > 1 and i < imax - 2 and tx > 1 and tx < dim_x - 2 and
      j > 1 and j < jmax - 2 and ty > 1 and ty < dim_y - 2)
  // clang-format on
  {
    const int tx2 = tx - 2;
    const int ty2 = ty - 1;

    int offset = elemOffset;
    Uout[offset] += (flux(tx2, ty2, ID, IY) - flux(tx2, ty2 + 1, ID, IY)) * dtdy;
    offset += arraySize;
    Uout[offset] += (flux(tx2, ty2, IP, IY) - flux(tx2, ty2 + 1, IP, IY)) * dtdy;
    offset += arraySize;
    Uout[offset] += (flux(tx2, ty2, IU, IY) - flux(tx2, ty2 + 1, IU, IY)) * dtdy;
    offset += arraySize;
    Uout[offset] += (flux(tx2, ty2, IV, IY) - flux(tx2, ty2 + 1, IV, IY)) * dtdy;
  }
  __syncthreads();

} // kernel_godunov_unsplit_2d_v3

#endif // KERNEL_GODUNOV_UNSPLIT_2D_V3_CUH_
