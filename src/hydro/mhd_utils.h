/**
 * \file mhd_utils.h
 * \brief Small MHD related utilities common to CPU / GPU code.
 *
 * These utility functions (find_speed_fast, etc...) are directly
 * adapted from Fortran original code found in RAMSES/DUMSES.
 *
 * \date 23 March 2011
 * \author Pierre Kestener.
 *
 * $Id: mhd_utils.h 3462 2014-06-25 10:12:53Z pkestene $
 */
#ifndef MHD_UTILS_H_
#define MHD_UTILS_H_

#include "constants.h"

/**
 * Compute the fast magnetosonic velocity.
 * 
 * IU is index to Vnormal
 * IA is index to Bnormal
 * 
 * IV, IW are indexes to Vtransverse1, Vtransverse2,
 * IB, IC are indexes to Btransverse1, Btransverse2
 *
 */
template <ComponentIndex3D dir>
__DEVICE__
real_riemann_t find_speed_fast(real_riemann_t qvar[NVAR_MHD])
{
   
  real_riemann_t d,p,a,b,c,b2,c2,d2,cf;

  d=qvar[ID]; p=qvar[IP]; 
  a=qvar[IA]; b=qvar[IB]; c=qvar[IC];

  b2 = a*a + b*b + c*c;
  c2 = gParams.gamma0 * p / d;
  d2 = HALF_F * (b2/d + c2);
  if (dir==IX)
    cf = SQRT( d2 + SQRT(d2*d2 - c2*a*a/d) );

  if (dir==IY)
    cf = SQRT( d2 + SQRT(d2*d2 - c2*b*b/d) );

  if (dir==IZ)
    cf = SQRT( d2 + SQRT(d2*d2 - c2*c*c/d) );

  return cf;
  
} // find_speed_fast

/**
 * Compute the Alfven velocity.
 *
 * The structure of qvar is :
 * rho, pressure, 
 * vnormal, vtransverse1, vtransverse2,
 * bnormal, btransverse1, btransverse2
 *
 */
 __DEVICE__
real_riemann_t find_speed_alfven(real_riemann_t qvar[NVAR_MHD])
{

  real_riemann_t d=qvar[ID];
  real_riemann_t a=qvar[IA];

  return SQRT(a*a/d);

} // find_speed_alfven

/**
 * Compute the Alfven velocity.
 *
 * Simpler interface.
 * \param[in] d density
 * \param[in] a normal magnetic field						\
 *
 */
 __DEVICE__
real_riemann_t find_speed_alfven(real_riemann_t d, real_riemann_t a)
{

  return SQRT(a*a/d);

} // find_speed_alfven

/**
 * Compute the 1d mhd fluxes from the conservative.
 *
 * Only used in Riemann solver HLL (probably cartesian only
 * compatible, since gas pressure is included).
 *
 * variables. The structure of qvar is : 
 * rho, pressure,
 * vnormal, vtransverse1, vtransverse2, 
 * bnormal, btransverse1, btransverse2.
 *
 * @param[in]  qvar state vector (primitive variables)
 * @param[out] cvar state vector (conservative variables)
 * @param[out] ff flux vector
 *
 */
 __DEVICE__
void find_mhd_flux(real_riemann_t qvar[NVAR_MHD], 
		   real_riemann_t (&cvar)[NVAR_MHD], 
		   real_riemann_t (&ff)[NVAR_MHD])
{

  // ISOTHERMAL
  real_t &cIso = ::gParams.cIso;
  real_riemann_t p;
  if (cIso>0) {
    // recompute pressure
    p = qvar[ID]*cIso*cIso;
  } else {
    p = qvar[IP];
  }
  // end ISOTHERMAL
  
  // local variables
  const real_riemann_t entho = ONE_F / (gParams.gamma0 - ONE_F);
  
  real_riemann_t d, u, v, w, a, b, c;
  d=qvar[ID]; 
  u=qvar[IU]; v=qvar[IV]; w=qvar[IW];
  a=qvar[IA]; b=qvar[IB]; c=qvar[IC];

  real_riemann_t ecin = HALF_F*(u*u+v*v+w*w)*d;
  real_riemann_t emag = HALF_F*(a*a+b*b+c*c);
  real_riemann_t etot = p*entho+ecin+emag;
  real_riemann_t ptot = p + emag;
  
  // compute conservative variables
  cvar[ID] = d;
  cvar[IP] = etot;
  cvar[IU] = d*u;
  cvar[IV] = d*v;
  cvar[IW] = d*w;
  cvar[IA] = a;
  cvar[IB] = b;
  cvar[IC] = c;

  // compute fluxes
  ff[ID] = d*u;
  ff[IP] = (etot+ptot)*u-a*(a*u+b*v+c*w);
  ff[IU] = d*u*u-a*a+ptot; /* *** WARNING pressure included *** */
  ff[IV] = d*u*v-a*b;
  ff[IW] = d*u*w-a*c;
  ff[IA] = ZERO_F;
  ff[IB] = b*u-a*v;
  ff[IC] = c*u-a*w;

} // find_mhd_flux

/**
 * Computes fast magnetosonic wave for each direction.
 *
 * \param[in]  qState       primitive variables state vector
 * \param[out] fastMagSpeed array containing fast magnetosonic speed along
 * x, y, and z direction.
 *
 * \tparam NDIM if NDIM==2, only computes magnetosonic speed along x
 * and y.
 */
template<DimensionType NDIM>
 __DEVICE__
void fast_mhd_speed(real_riemann_t qState[NVAR_MHD], real_riemann_t (&fastMagSpeed)[3])
{

  real_riemann_t& rho = qState[ID];
  real_riemann_t& p   = qState[IP];
  /*real_riemann_t& vx  = qState[IU];
    real_riemann_t& vy  = qState[IV];
    real_riemann_t& vz  = qState[IW];*/
  real_riemann_t& bx  = qState[IA];
  real_riemann_t& by  = qState[IB];
  real_riemann_t& bz  = qState[IC];

  real_riemann_t mag_perp,alfv,vit_son,som_vit,som_vit2,delta,fast_speed;

  // compute fast magnetosonic speed along X
  mag_perp =  (by*by + bz*bz)  /  rho;  // bt ^2 / rho
  alfv     =   bx*bx           /  rho;  // bx / sqrt(4pi*rho)
  vit_son  =  gParams.gamma0*p /  rho;  // sonic contribution :  gamma*P / rho
  
  som_vit  =  mag_perp + alfv + vit_son ; // whatever direction,
					  // always the same
  som_vit2 =  som_vit * som_vit;

  delta    = FMAX(ZERO_F, som_vit2 - 4 * vit_son*alfv ); 
  
  fast_speed =  HALF_F * ( som_vit + SQRT( delta ) );
  fast_speed =  SQRT( fast_speed );

  fastMagSpeed[IX] = fast_speed;

  // compute fast magnetosonic speed along Y
  mag_perp =  (bx*bx + bz*bz)  /  rho;  
  alfv     =   by*by           /  rho;  
  
  delta    = FMAX(ZERO_F, som_vit2 - 4 * vit_son*alfv ); 
  
  fast_speed =  HALF_F * ( som_vit + SQRT( delta ) );
  fast_speed =  SQRT( fast_speed );

  fastMagSpeed[IY] = fast_speed;

  // compute fast magnetosonic speed along Z
  if (NDIM == THREE_D) {
    mag_perp =  (bx*bx + by*by)  /  rho;  
    alfv     =   bz*bz           /  rho;  
    
    delta    = FMAX(ZERO_F, som_vit2 - 4 * vit_son*alfv ); 
    
    fast_speed =  HALF_F * ( som_vit + SQRT( delta ) );
    fast_speed =  SQRT( fast_speed );

    fastMagSpeed[IZ] = fast_speed;
  }

} // fast_mhd_speed

/**
 * Computes fastest signal speed for each direction.
 *
 * \param[in]  qState       primitive variables state vector
 * \param[out] fastInfoSpeed array containing fastest information speed along
 * x, y, and z direction.
 *
 * Directionnal information speed being defined as :
 * directionnal fast magneto speed + FABS(velocity component)
 *
 * \warning This routine uses gamma ! You need to set gamma to something very near to 1
 *
 * \tparam NDIM if NDIM==2, only computes information speed along x
 * and y.
 */
template<DimensionType NDIM>
 __DEVICE__
void find_speed_info(real_riemann_t qState[NVAR_MHD], 
		     real_riemann_t (&fastInfoSpeed)[3])
{

  real_riemann_t d,p,a,b,c,b2,c2,d2,cf;
  real_riemann_t &u = qState[IU];
  real_riemann_t &v = qState[IV];
  real_riemann_t &w = qState[IW];

  d=qState[ID]; p=qState[IP]; 
  a=qState[IA]; b=qState[IB]; c=qState[IC];

  /*
   * compute fastest info speed along X
   */

  // square norm of magnetic field
  b2 = a*a + b*b + c*c;

  // square speed of sound
  c2 = gParams.gamma0 * p / d;

  d2 = HALF_F * (b2/d + c2);

  cf = SQRT( d2 + SQRT(d2*d2 - c2*a*a/d) );

  fastInfoSpeed[IX] = cf+FABS(u);

  // compute fastest info speed along Y
  cf = SQRT( d2 + SQRT(d2*d2 - c2*b*b/d) );

  fastInfoSpeed[IY] = cf+FABS(v);

  
  // compute fastest info speed along Z
  if (NDIM == THREE_D) {
    cf = SQRT( d2 + SQRT(d2*d2 - c2*c*c/d) );
    
    fastInfoSpeed[IZ] = cf+FABS(w);
  } // end THREE_D

} // find_speed_info

/**
 * Computes fastest signal speed for each direction.
 *
 * \param[in]  qState       primitive variables state vector
 * \param[out] fastInfoSpeed fastest information speed along x
 *
 * \warning This routine uses gamma ! You need to set gamma to something very near to 1
 *
 */
 __DEVICE__
real_riemann_t find_speed_info(real_riemann_t qState[NVAR_MHD])
{

  real_riemann_t d,p,a,b,c,b2,c2,d2,cf;
  real_riemann_t &u = qState[IU];
  //real_riemann_t &v = qState[IV];
  //real_riemann_t &w = qState[IW];

  d=qState[ID]; p=qState[IP]; 
  a=qState[IA]; b=qState[IB]; c=qState[IC];

  // compute fastest info speed along X
  b2 = a*a + b*b + c*c;
  c2 = gParams.gamma0 * p / d;
  d2 = HALF_F * (b2/d + c2);
  cf = SQRT( d2 + SQRT(d2*d2 - c2*a*a/d) );

  // return value
  return cf+FABS(u);

} // find_speed_info

#endif // MHD_UTILS_H_
