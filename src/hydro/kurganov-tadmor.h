/**
 * \file kurganov-tadmor.h
 * \brief Intermediate routines used in the Kurganov-Tadmor scheme.
 *
 * Implement small CPU routine working on small local data (and their
 * GPU __DEVICE__ counterparts)  
 *
 * \date 05/02/2010
 * \author Pierre Kestener
 *
 * $Id: kurganov-tadmor.h 1784 2012-02-21 10:34:58Z pkestene $
 *
 */
#ifndef KURGANOV_TADMOR_H_
#define KURGANOV_TADMOR_H_

#include "real_type.h"
#include "constants.h"

////////////////////////////////////////////////////////////////////////////
/** 
 * \fn void spectral_radii(real_t u[NVAR], real_t& rx, real_t& ry)
 * \brief return the maximun absolute value of Roe Matrix eigenvalue.
 * 
 * 
 * @param u : input NVAR-dimensional vector 
 * @param rx
 * @param ry
 */
template <NvarSimulation NVAR>
inline __DEVICE__
void spectral_radii(real_t u[NVAR], real_t& rx, real_t& ry)
{
  
  real_t rho = FMAX( u[ID], ::gParams.smallr);
  real_t vx  = u[IU]/rho;
  real_t vy  = u[IV]/rho;
  real_t p   = FMAX( (::gParams.gamma0-1.0)*(u[IP]-0.5*rho*(vx*vx + vy*vy)), ::gParams.smallp);
  real_t c   = SQRT(::gParams.gamma0*p/rho);
  rx        = FABS(vx)+c;
  ry        = FABS(vy)+c;
} // spectral_radii


/**
 * return the smallest absolute value between a and b (if a
 * and b are of same sign).
 * This was originally implement as .5*(sign(x) + sign(y))*min(fabs(x),fabs(y));
 */
template<typename T> inline __DEVICE__
T minmod(const T& a, T b)
{
  return a*b<=0?0:(a>0?(a<b?a:b):(a<b?b:a));
} // minmod

template<typename T> inline __DEVICE__
T minmod3(const T& a, const T& b, const T& c)
{
  return minmod(a, minmod(b,c));
} // minmod3


/** 
 * compute fluxes along direction x and y (Euler equation, gas dynamics)
 * 
 * @param u : conservative variables
 */
template <NvarSimulation NVAR>
__DEVICE__ void get_flux(real_t u[NVAR], real_t (&fx)[NVAR], real_t (&fy)[NVAR])
{
  real_t p = FMAX( (::gParams.gamma0-1.0)*(u[IP]-0.5*(u[IU]*u[IU]+u[IV]*u[IV])/u[ID]), u[ID] * ::gParams.smallp);
  
  fx[ID]=u[IU];
  fx[IU]=(u[IU]*u[IU])/u[ID] + p;
  fx[IV]= u[IU]*u[IV]/u[ID];
  fx[IP]=(u[IP]+p)*(u[IU]/u[ID]);
  
  fy[ID]=u[IV];
  fy[IU]=(u[IU]*u[IV]/u[ID]);
  fy[IV]=(u[IV]*u[IV])/u[ID] + p;
  fy[IP]=(u[IP]+p)*(u[IV]/u[ID]);
} // get_flux


#endif // KURGANOV_TADMOR_H_
