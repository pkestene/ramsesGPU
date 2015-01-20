/**
 * \file relaxingTVD.h
 * \brief Some utilities used either in the CPU or GPU version of the relaxing TVD scheme.
 *
 * \date 25-Jan-2011
 * \author P. Kestener 
 *
 * $Id: relaxingTVD.h 1784 2012-02-21 10:34:58Z pkestene $
 */
#ifndef RELAXING_TVD_H_
#define RELAXING_TVD_H_

#include "real_type.h"
#include "constants.h"

/**
 * calculate cell-centered fluxes and freezing speed.
 * Take care to swap u components when doing sweep along direction Y or Z !!!
 *
 * \param[in]  u : current conservative variables state
 * \param[out] w : returned flux
 * \param[out] c : returned freezing speed
 */
template <NvarSimulation NVAR>
__DEVICE__
void averageFlux(real_t u[NVAR],
		 real_t (&w)[NVAR],
		 real_t &c)
{
  real_t rho = FMAX(u[ID], 1e-9);
  real_t v = u[IU]/rho;
  real_t ek;
  if (NVAR == NVAR_2D) {
    ek =  HALF_F * ( u[IU]*u[IU] + u[IV]*u[IV] ) / rho;
  } else { // 3D
    ek =  HALF_F * ( u[IU]*u[IU] + u[IV]*u[IV] + u[IW]*u[IW] ) / rho;
  }
  real_t P = FMAX(ZERO_F, (gParams.gamma0-1)*(u[IP]-ek));
  c = FABS(v) + FMAX( SQRT(gParams.gamma0*P/rho),1e-5);
  w[ID]=rho*v;
  w[IU]=(u[IU]*v+P);
  w[IV]=u[IV]*v;
  if (NVAR == NVAR_3D)
    w[IW]=u[IW]*v;
  w[IP]=(u[IP]+P)*v;

} // averageFlux

/**
 * Van Leer type flux limiter.
 *
 * param[out] f : flux
 * param[in]  a : delta flux
 * param[in]  b : delta flux
 */
__DEVICE__
void vanleer(real_t (&f),
	     real_t    a,
	     real_t    b)
{
  real_t c;
  
  c = a*b;
    
  if (c > 0)
    f += 2*c/(a+b);

} // vanleer

/**
 * minmod type flux limiter.
 *
 * param[out] f : flux
 * param[in]  a : delta flux
 * param[in]  b : delta flux
 */
__DEVICE__
void minmod(real_t (&f),
	    real_t    a,
	    real_t    b)
{
  f += (COPYSIGN(ONE_F,a) + COPYSIGN(ONE_F,b)) * FMIN(FABS(a),FABS(b))/2.;
} // minmod


/**
 * superbee type flux limiter.
 *
 * param[out] f : flux
 * param[in]  a : delta flux
 * param[in]  b : delta flux
 */
__DEVICE__
void superbee(real_t (&f),
	     real_t    a,
	     real_t    b)
{
  if ( FABS(1.) > FABS(b) ) {
    f += (COPYSIGN(1.,a)+COPYSIGN(1.,b))*FMIN(FABS(a),abs(2*b))/2.;
  } else {
    f += (COPYSIGN(1.,a)+COPYSIGN(1.,b))*FMIN(FABS(2*a),abs(b))/2.;
  }
} // superbee

#endif /* RELAXING_TVD_H_ */
