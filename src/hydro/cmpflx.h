/**
 * \file cmpflx.h
 * \brief Implements the CPU/GPU device routines to compute fluxes update
 * from the Godunov state.
 *
 * \author F. Chateau
 *
 * $Id: cmpflx.h 1784 2012-02-21 10:34:58Z pkestene $
 */ 
#ifndef CMPFLX_H_
#define CMPFLX_H_

#include "real_type.h"
#include "constants.h"

/**
 * Compute cell fluxes from the Godunov state
 * @param qgdnv input Godunov state
 * @param flux output flux vector
 */
template <NvarSimulation NVAR>
__DEVICE__ 
void cmpflx(real_t qgdnv[NVAR], real_t (&flux)[NVAR])
{
  // Compute fluxes
  // Mass density
  flux[ID] = qgdnv[ID] * qgdnv[IU];
  
  // Normal momentum
  flux[IU] = flux[ID] * qgdnv[IU] + qgdnv[IP];
  
  // Transverse momentum 1
  flux[IV] = flux[ID] * qgdnv[IV];

  // Transverse momentum 2
  if (NVAR == NVAR_3D)
    flux[IW] = flux[ID] * qgdnv[IW];
  
  // Total energy
  real_t entho = (real_t) 1.0f / (gParams.gamma0 - 1.0f);
  real_t ekin;
  if (NVAR == NVAR_3D) {
    ekin = 0.5f * qgdnv[ID] * (qgdnv[IU]*qgdnv[IU] + qgdnv[IV]*qgdnv[IV] + qgdnv[IW]*qgdnv[IW]);
  } else {
    ekin = 0.5f * qgdnv[ID] * (qgdnv[IU]*qgdnv[IU] + qgdnv[IV]*qgdnv[IV]);
  }
  real_t etot = qgdnv[IP] * entho + ekin;
  flux[IP] = qgdnv[IU] * (etot + qgdnv[IP]);
}

/*
__DEVICE__ 
void cmpflx(double qgdnv[NVAR], double (&flux)[NVAR])
{
  // Compute fluxes
  // Mass density
  flux[ID] = qgdnv[ID] * qgdnv[IU];
  
  // Normal momentum
  flux[IU] = flux[ID] * qgdnv[IU] + qgdnv[IP];
  
  // Transverse momentum 1
  flux[IV] = flux[ID] * qgdnv[IV];
  
  // Total energy
  double entho = 1.0f / (gamma0 - 1.0f);
  double ekin = 0.5f * qgdnv[ID] * (qgdnv[IU]*qgdnv[IU] + qgdnv[IV]*qgdnv[IV]);
  double etot = qgdnv[IP] * entho + ekin;
  flux[IP] = qgdnv[IU] * (etot + qgdnv[IP]);
}
*/
#endif /*CMPFLX_H_*/
