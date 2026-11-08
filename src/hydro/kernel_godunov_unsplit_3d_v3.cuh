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
 * \file kernel_godunov_unsplit_3d_v3.cuh
 * \brief Define version 4 of godunov kernel.
 */
#ifndef KERNEL_GODUNOV_UNSPLIT_3D_V3_CUH_
#define KERNEL_GODUNOV_UNSPLIT_3D_V3_CUH_

#include "real_type.h"
#include "constants.h"
#include "kernel_godunov_unsplit_3d_v3_utils.cuh"

/**
 * Unsplit Godunov kernel for 3D data (version 4).
 *
 * Primitive variables are computed here.
 *
 */
__global__ void
kernel_godunov_unsplit_3d_v3(const real_t * __restrict__ Uin,
                             real_t * __restrict__ Uout,
                             int    pitch,
                             int    imax,
                             int    jmax,
                             int    kmax,
                             real_t dtdx,
                             real_t dtdy,
                             real_t dtdz,
                             real_t dt,
                             bool   gravityEnabled)
{
  // Block index
  auto const & bx = blockIdx.x;
  auto const & by = blockIdx.y;
  auto const & bz = blockIdx.z;

  // block dimension
  auto const & dim_x = blockDim.x;
  auto const & dim_y = blockDim.y;
  auto const & dim_z = blockDim.z;

  // Thread index
  auto const & tx = threadIdx.x;
  auto const & ty = threadIdx.y;
  auto const & tz = threadIdx.z;

  const int i = __mul24(bx, dim_x - 4) + tx;
  const int j = __mul24(by, dim_y - 4) + ty;
  const int k = __mul24(bz, dim_z - 4) + tz;

  const int arraySize = pitch * jmax * kmax;
  const int elemOffset = i + pitch * j + pitch * jmax * k;

  extern __shared__ char storage[];

  void * qprim_data_ptr = reinterpret_cast<void *>(&storage);
  void * qm_data_ptr =
    reinterpret_cast<void *>(&storage[QPrim3D::required_size_in_bytes(dim_x, dim_y, dim_z)]);
  void * qp_data_ptr =
    reinterpret_cast<void *>(&storage[QPrim3D::required_size_in_bytes(dim_x, dim_y, dim_z) +
                                      QRecons3D::required_size_in_bytes(dim_x, dim_y, dim_z)]);
  void * flux_data_ptr =
    reinterpret_cast<void *>(&storage[QPrim3D::required_size_in_bytes(dim_x, dim_y, dim_z) +
                                      2 * QRecons3D::required_size_in_bytes(dim_x, dim_y, dim_z)]);

  // static allocation of all containers (this will call default constructor that does nothing)
  __shared__ QPrim3D   q;
  __shared__ QRecons3D qm;
  __shared__ QRecons3D qp;
  __shared__ QRecons3D flux;

  // call constructor at designated address with placement new (only 1 thread in block)
  if (tx == 0 and ty == 0 and tz == 0)
  {
    ::new (&q) QPrim3D(qprim_data_ptr, dim_x, dim_y, dim_z);
    ::new (&qm) QRecons3D(qm_data_ptr, dim_x, dim_y, dim_z);
    ::new (&qp) QRecons3D(qp_data_ptr, dim_x, dim_y, dim_z);
    ::new (&flux) QRecons3D(flux_data_ptr, dim_x, dim_y, dim_z);
  }
  __syncthreads();

  // conservative variables
  real_t uIn[NVAR_3D];

  real_t * gravin = gParams.arrayList[A_GRAV];

  // compute primitive variables
  if (i >= 0 and i < imax and j >= 0 and j < jmax and k >= 0 and k < kmax)
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
    offset += arraySize;
    uIn[IW] = Uin[offset];

    // Convert to primitive variables
    constoprim_3D(uIn, q, tx, ty, tz, c);
  }
  __syncthreads();

  //
  // compute reconstructed on X faces
  //

  // clang-format off
  if (i > 0 and i < imax - 1 and tx > 0 and tx < dim_x - 1 and
      j > 0 and j < jmax - 1 and ty > 1 and ty < dim_y - 2 and
      k > 0 and k < kmax - 1 and tz > 1 and tz < dim_z - 2)
  // clang-format on
  {

    // compute left and right state
    trace_unsplit_hydro_3d_by_direction_shmem(q, tx, ty, tz, dtdx, dtdy, dtdz, IX, qm, qp);

    if (gravityEnabled)
    {
      const int tx2 = tx - 1;
      const int ty2 = ty - 2;
      const int tz2 = tz - 2;

      // we need to modify input to flux computation with
      // gravity predictor (half time step)
      qm(tx2, ty2, tz2, IU, IX) += HALF_F * dt * gravin[elemOffset + IX * arraySize];
      qm(tx2, ty2, tz2, IV, IX) += HALF_F * dt * gravin[elemOffset + IY * arraySize];
      qm(tx2, ty2, tz2, IW, IX) += HALF_F * dt * gravin[elemOffset + IZ * arraySize];

      qp(tx2, ty2, tz2, IU, IX) += HALF_F * dt * gravin[elemOffset + IX * arraySize];
      qp(tx2, ty2, tz2, IV, IX) += HALF_F * dt * gravin[elemOffset + IY * arraySize];
      qp(tx2, ty2, tz2, IW, IX) += HALF_F * dt * gravin[elemOffset + IZ * arraySize];

    } // end gravityEnabled
  }
  __syncthreads();

  //
  // Compute fluxes along X
  //

  // clang-format off
  if (i > 1 and i < imax - 1 and tx > 1 and tx < dim_x - 1 and
      j > 1 and j < jmax - 1 and ty > 1 and ty < dim_y - 2 and
      k > 1 and k < kmax - 1 and tz > 1 and tz < dim_z - 2)
  // clang-format on
  {
    const int tx2 = tx - 1;
    const int ty2 = ty - 2;
    const int tz2 = tz - 2;

    // Solve Riemann problem at X-interfaces and compute X-fluxes, and output the fluxes
    riemann<3, IX>(qm, qp, tx2, ty2, tz2, flux);
  }
  __syncthreads();

  //
  // Update along X
  //

  // clang-format off
  if (i > 1 and i < imax - 2 and tx > 1 and tx < dim_x - 2 and
      j > 1 and j < jmax - 2 and ty > 1 and ty < dim_y - 2 and
      k > 1 and k < kmax - 2 and tz > 1 and tz < dim_z - 2)
  // clang-format on
  {
    const int tx2 = tx - 1;
    const int ty2 = ty - 2;
    const int tz2 = tz - 2;

    int offset = elemOffset;
    Uout[offset] += (flux(tx2, ty2, tz2, ID, IX) - flux(tx2 + 1, ty2, tz2, ID, IX)) * dtdx;
    offset += arraySize;
    Uout[offset] += (flux(tx2, ty2, tz2, IP, IX) - flux(tx2 + 1, ty2, tz2, IP, IX)) * dtdx;
    offset += arraySize;
    Uout[offset] += (flux(tx2, ty2, tz2, IU, IX) - flux(tx2 + 1, ty2, tz2, IU, IX)) * dtdx;
    offset += arraySize;
    Uout[offset] += (flux(tx2, ty2, tz2, IV, IX) - flux(tx2 + 1, ty2, tz2, IV, IX)) * dtdx;
    offset += arraySize;
    Uout[offset] += (flux(tx2, ty2, tz2, IW, IX) - flux(tx2 + 1, ty2, tz2, IW, IX)) * dtdx;
  }
  __syncthreads();

  //
  // compute reconstructed on Y faces
  //

  // clang-format off
  if (i > 0 and i < imax - 1 and tx > 1 and tx < dim_x - 2 and
      j > 0 and j < jmax - 1 and ty > 0 and ty < dim_y - 1 and
      k > 0 and k < kmax - 1 and tz > 1 and tz < dim_z - 2)
  // clang-format on
  {

    // compute left and right state
    trace_unsplit_hydro_3d_by_direction_shmem(q, tx, ty, tz, dtdx, dtdy, dtdz, IY, qm, qp);

    if (gravityEnabled)
    {
      const int tx2 = tx - 2;
      const int ty2 = ty - 1;
      const int tz2 = tz - 2;

      // we need to modify input to flux computation with
      // gravity predictor (half time step)
      qm(tx2, ty2, tz2, IU, IY) += HALF_F * dt * gravin[elemOffset + IX * arraySize];
      qm(tx2, ty2, tz2, IV, IY) += HALF_F * dt * gravin[elemOffset + IY * arraySize];
      qm(tx2, ty2, tz2, IW, IY) += HALF_F * dt * gravin[elemOffset + IZ * arraySize];

      qp(tx2, ty2, tz2, IU, IY) += HALF_F * dt * gravin[elemOffset + IX * arraySize];
      qp(tx2, ty2, tz2, IV, IY) += HALF_F * dt * gravin[elemOffset + IY * arraySize];
      qp(tx2, ty2, tz2, IW, IY) += HALF_F * dt * gravin[elemOffset + IZ * arraySize];

    } // end gravityEnabled
  }
  __syncthreads();

  //
  // Compute fluxes along Y
  //

  // clang-format off
  if (i > 1 and i < imax - 1 and tx > 1 and tx < dim_x - 2 and
      j > 1 and j < jmax - 1 and ty > 1 and ty < dim_y - 1 and
      k > 1 and k < kmax - 1 and tz > 1 and tz < dim_z - 2)
  // clang-format on
  {
    const int tx2 = tx - 2;
    const int ty2 = ty - 1;
    const int tz2 = tz - 2;

    // Solve Riemann problem at X-interfaces and compute X-fluxes, and output the fluxes
    riemann<3, IY>(qm, qp, tx2, ty2, tz2, flux);
  }
  __syncthreads();

  //
  // Update along Y
  //

  // clang-format off
  if (i > 1 and i < imax - 2 and tx > 1 and tx < dim_x - 2 and
      j > 1 and j < jmax - 2 and ty > 1 and ty < dim_y - 2 and
      k > 1 and k < kmax - 2 and tz > 1 and tz < dim_z - 2)
  // clang-format on
  {
    const int tx2 = tx - 2;
    const int ty2 = ty - 1;
    const int tz2 = tz - 2;

    int offset = elemOffset;
    Uout[offset] += (flux(tx2, ty2, tz2, ID, IY) - flux(tx2, ty2 + 1, tz2, ID, IY)) * dtdy;
    offset += arraySize;
    Uout[offset] += (flux(tx2, ty2, tz2, IP, IY) - flux(tx2, ty2 + 1, tz2, IP, IY)) * dtdy;
    offset += arraySize;
    Uout[offset] += (flux(tx2, ty2, tz2, IU, IY) - flux(tx2, ty2 + 1, tz2, IU, IY)) * dtdy;
    offset += arraySize;
    Uout[offset] += (flux(tx2, ty2, tz2, IV, IY) - flux(tx2, ty2 + 1, tz2, IV, IY)) * dtdy;
    offset += arraySize;
    Uout[offset] += (flux(tx2, ty2, tz2, IW, IY) - flux(tx2, ty2 + 1, tz2, IW, IY)) * dtdy;
  }
  __syncthreads();

  //
  // compute reconstructed on Z faces
  //

  // clang-format off
  if (i > 0 and i < imax - 1 and tx > 1 and tx < dim_x - 2 and
      j > 0 and j < jmax - 1 and ty > 1 and ty < dim_y - 2 and
      k > 0 and k < kmax - 1 and tz > 0 and tz < dim_z - 1)
  // clang-format on
  {

    // compute left and right state
    trace_unsplit_hydro_3d_by_direction_shmem(q, tx, ty, tz, dtdx, dtdy, dtdz, IZ, qm, qp);

    if (gravityEnabled)
    {
      const int tx2 = tx - 2;
      const int ty2 = ty - 2;
      const int tz2 = tz - 1;

      // we need to modify input to flux computation with
      // gravity predictor (half time step)
      qm(tx2, ty2, tz2, IU, IZ) += HALF_F * dt * gravin[elemOffset + IX * arraySize];
      qm(tx2, ty2, tz2, IV, IZ) += HALF_F * dt * gravin[elemOffset + IY * arraySize];
      qm(tx2, ty2, tz2, IW, IZ) += HALF_F * dt * gravin[elemOffset + IZ * arraySize];

      qp(tx2, ty2, tz2, IU, IZ) += HALF_F * dt * gravin[elemOffset + IX * arraySize];
      qp(tx2, ty2, tz2, IV, IZ) += HALF_F * dt * gravin[elemOffset + IY * arraySize];
      qp(tx2, ty2, tz2, IW, IZ) += HALF_F * dt * gravin[elemOffset + IZ * arraySize];

    } // end gravityEnabled
  }
  __syncthreads();

  //
  // Compute fluxes along Z
  //

  // clang-format off
  if (i > 1 and i < imax - 1 and tx > 1 and tx < dim_x - 2 and
      j > 1 and j < jmax - 1 and ty > 1 and ty < dim_y - 2 and
      k > 1 and k < kmax - 1 and tz > 1 and tz < dim_z - 1)
  // clang-format on
  {
    const int tx2 = tx - 2;
    const int ty2 = ty - 2;
    const int tz2 = tz - 1;

    // Solve Riemann problem at X-interfaces and compute X-fluxes, and output the fluxes
    riemann<3, IZ>(qm, qp, tx2, ty2, tz2, flux);
  }
  __syncthreads();

  //
  // Update along Z
  //

  // clang-format off
  if (i > 1 and i < imax - 2 and tx > 1 and tx < dim_x - 2 and
      j > 1 and j < jmax - 2 and ty > 1 and ty < dim_y - 2 and
      k > 1 and k < kmax - 2 and tz > 1 and tz < dim_z - 2)
  // clang-format on
  {
    const int tx2 = tx - 2;
    const int ty2 = ty - 2;
    const int tz2 = tz - 1;

    int offset = elemOffset;
    Uout[offset] += (flux(tx2, ty2, tz2, ID, IZ) - flux(tx2, ty2, tz2 + 1, ID, IZ)) * dtdz;
    offset += arraySize;
    Uout[offset] += (flux(tx2, ty2, tz2, IP, IZ) - flux(tx2, ty2, tz2 + 1, IP, IZ)) * dtdz;
    offset += arraySize;
    Uout[offset] += (flux(tx2, ty2, tz2, IU, IZ) - flux(tx2, ty2, tz2 + 1, IU, IZ)) * dtdz;
    offset += arraySize;
    Uout[offset] += (flux(tx2, ty2, tz2, IV, IZ) - flux(tx2, ty2, tz2 + 1, IV, IZ)) * dtdz;
    offset += arraySize;
    Uout[offset] += (flux(tx2, ty2, tz2, IW, IZ) - flux(tx2, ty2, tz2 + 1, IW, IZ)) * dtdz;
  }
  __syncthreads();

} // kernel_godunov_unsplit_3d_v3

#endif // KERNEL_GODUNOV_UNSPLIT_3D_V3_CUH_
