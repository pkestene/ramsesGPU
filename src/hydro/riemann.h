/**
 * \file riemann.h
 * \brief Provides CPU/GPU riemann solver routines.
 *
 * Implement different kinds of Riemann solvers
 * adapted from original fortran code found in RAMSES .
 * See file godunov_utils.f90 in RAMSES.
 *
 * \author F. Chateau and P. Kestener
 *
 * $Id: riemann.h 3452 2014-06-17 10:09:48Z pkestene $
 */
#ifndef RIEMANN_H_
#define RIEMANN_H_

#include "real_type.h"
#include "constants.h"
#include "cmpflx.h"

/** 
 * Riemann solver, equivalent to riemann_approx in RAMSES (see file
 * godunov_utils.f90 in RAMSES).
 * 
 * @param[in] qleft  : input left state
 * @param[in] qright : input right state
 * @param[out] qgdnv : output Godunov state
 * @param[out] flux  : output flux
 */
template <NvarSimulation NVAR>
__DEVICE__
void riemann_approx(real_t qleft[NVAR], real_t qright[NVAR], 
		    real_t (&qgdnv)[NVAR], real_t (&flux)[NVAR])
{
	// Pressure, density and velocity
	real_t rl = FMAX(qleft [ID], gParams.smallr);
	real_t ul =      qleft [IU];
	real_t pl = FMAX(qleft [IP], rl*gParams.smallp);
	real_t rr = FMAX(qright[ID], gParams.smallr);
	real_t ur =      qright[IU];
	real_t pr = FMAX(qright[IP], rr*gParams.smallp);

	// Lagrangian sound speed
	real_t cl = gParams.gamma0*pl*rl;
	real_t cr = gParams.gamma0*pr*rr;

	// First guess
	real_t wl = SQRT(cl);
	real_t wr = SQRT(cr);
	real_t pstar = FMAX(((wr*pl+wl*pr)+wl*wr*(ul-ur))/(wl+wr), (real_t) ZERO_F);
	real_t pold = pstar;
	real_t conv = ONE_F;

	// Newton-Raphson iterations to find pstar at the required accuracy
	for(int iter = 0; iter < gParams.niter_riemann and conv > 1e-6; ++iter)
	{
		real_t wwl = SQRT(cl*(ONE_F+gParams.gamma6*(pold-pl)/pl));
		real_t wwr = SQRT(cr*(ONE_F+gParams.gamma6*(pold-pr)/pr));
		real_t ql = 2.0f*wwl*wwl*wwl/(wwl*wwl+cl);
		real_t qr = 2.0f*wwr*wwr*wwr/(wwr*wwr+cr);
		real_t usl = ul-(pold-pl)/wwl;
		real_t usr = ur+(pold-pr)/wwr;
		real_t delp = FMAX(qr*ql/(qr+ql)*(usl-usr),-pold);

		pold = pold+delp;
		conv = FABS(delp/(pold+gParams.smallpp));	 // Convergence indicator
	}

	// Star region pressure
	// for a two-shock Riemann problem
	pstar = pold;
	wl = SQRT(cl*(ONE_F+gParams.gamma6*(pstar-pl)/pl));
	wr = SQRT(cr*(ONE_F+gParams.gamma6*(pstar-pr)/pr));

	// Star region velocity
	// for a two shock Riemann problem
	real_t ustar = HALF_F * (ul + (pl-pstar)/wl + ur - (pr-pstar)/wr);
	
	// Left going or right going contact wave
	real_t sgnm = COPYSIGN(ONE_F, ustar);
	
	// Left or right unperturbed state
	real_t ro, uo, po, wo;
	if(sgnm > ZERO_F)
	{
		ro = rl;
		uo = ul;
		po = pl;
		wo = wl;
	}
	else
	{
		ro = rr;
		uo = ur;
		po = pr;
		wo = wr;
	}
	real_t co = FMAX(gParams.smallc, SQRT(FABS(gParams.gamma0*po/ro)));

	// Star region density (Shock, FMAX prevents vacuum formation in star region)
	real_t rstar = FMAX((real_t) (ro/(ONE_F+ro*(po-pstar)/(wo*wo))), (real_t) (gParams.smallr));
	// Star region sound speed
	real_t cstar = FMAX(gParams.smallc, SQRT(FABS(gParams.gamma0*pstar/rstar)));

	// Compute rarefaction head and tail speed
	real_t spout  = co    - sgnm*uo;
	real_t spin   = cstar - sgnm*ustar;
	// Compute shock speed
	real_t ushock = wo/ro - sgnm*uo;

	if(pstar >= po)
	{
		spin  = ushock;
		spout = ushock;
	}

	// Sample the solution at x/t=0
	real_t scr = FMAX(spout-spin, gParams.smallc+FABS(spout+spin));
	real_t frac = HALF_F * (ONE_F + (spout + spin)/scr);
	//real_t frac = SATURATE();
	if (isnan(frac))
	  frac = ZERO_F;
	else
	  frac = SATURATE(frac);

	qgdnv[ID] = frac*rstar + (ONE_F-frac)*ro;
	qgdnv[IU] = frac*ustar + (ONE_F-frac)*uo;
	qgdnv[IP] = frac*pstar + (ONE_F-frac)*po;

	if(spout < ZERO_F)
	{
		qgdnv[ID] = ro;
		qgdnv[IU] = uo;
		qgdnv[IP] = po;
	}

	if(spin > ZERO_F)
	{
		qgdnv[ID] = rstar;
		qgdnv[IU] = ustar;
		qgdnv[IP] = pstar;
	}

	// transverse velocity
	if(sgnm > ZERO_F)
	{
		qgdnv[IV] = qleft[IV];
		if (NVAR == NVAR_3D)
		  qgdnv[IW] = qleft[IW];
	}
	else
	{
		qgdnv[IV] = qright[IV];
		if (NVAR == NVAR_3D)
		  qgdnv[IW] = qright[IW];
	}

	cmpflx<NVAR>(qgdnv, flux);

} // riemann_approx

/** 
 * Riemann solver, equivalent to riemann_hll in RAMSES (see file
 * godunov_utils.f90 in RAMSES).
 * 
 * This is the HYDRO only version. The MHD version is in file riemann_mhd.h
 *
 * Reference : E.F. Toro, Riemann solvers and numerical methods for
 * fluid dynamics, Springer, chapter 10 (The HLL and HLLC Riemann solver).
 *
 * @param[in] qleft  : input left state
 * @param[in] qright : input right state
 * @param[out] qgdnv : output Godunov state
 * @param[out] flux  : output flux
 */
template <NvarSimulation NVAR>
__DEVICE__
void riemann_hll(real_t qleft[NVAR], real_t qright[NVAR], 
		 real_t (&qgdnv)[NVAR], real_t (&flux)[NVAR])
{

  // 1D HLL Riemann solver
  
  // constants
  //const real_t smallp = gParams.smallc*gParams.smallc/gParams.gamma0;
  const real_t entho = ONE_F / (gParams.gamma0 - ONE_F);

  // Maximum wave speed
  real_t rl=FMAX(qleft [ID],gParams.smallr);
  real_t ul=     qleft [IU];
  real_t pl=FMAX(qleft [IP],rl*gParams.smallp);

  real_t rr=FMAX(qright[ID],gParams.smallr);
  real_t ur=     qright[IU];
  real_t pr=FMAX(qright[IP],rr*gParams.smallp);

  real_t cl= SQRT(gParams.gamma0*pl/rl);
  real_t cr= SQRT(gParams.gamma0*pr/rr);

  real_t SL = FMIN(FMIN(ul,ur)-FMAX(cl,cr),(real_t) ZERO_F);
  real_t SR = FMAX(FMAX(ul,ur)+FMAX(cl,cr),(real_t) ZERO_F);

  // Compute average velocity
  qgdnv[IU] = HALF_F*(qleft[IU]+qright[IU]);
  
  // Compute conservative variables
  real_t uleft[NVAR], uright[NVAR];
  uleft [ID] = qleft [ID];
  uright[ID] = qright[ID];
  uleft [IP] = qleft [IP]*entho + HALF_F*qleft [ID]*qleft [IU]*qleft [IU];
  uright[IP] = qright[IP]*entho + HALF_F*qright[ID]*qright[IU]*qright[IU];
  uleft [IP] += HALF_F*qleft [ID]*qleft [IV]*qleft [IV];
  uright[IP] += HALF_F*qright[ID]*qright[IV]*qright[IV];
  if (NVAR == NVAR_3D) {
    uleft [IP] += HALF_F*qleft [ID]*qleft [IW]*qleft [IW];
    uright[IP] += HALF_F*qright[ID]*qright[IW]*qright[IW];
  }
  uleft [IU] = qleft [ID]*qleft [IU];
  uright[IU] = qright[ID]*qright[IU];

  // Other advected quantities
  uleft [IV] = qleft [ID]*qleft [IV];
  uright[IV] = qright[ID]*qright[IV];
  if (NVAR == NVAR_3D) {
    uleft [IW] = qleft [ID]*qleft [IW];
    uright[IW] = qright[ID]*qright[IW];
  }

  // Compute left and right fluxes
  real_t fleft[NVAR], fright[NVAR];
  fleft [ID] = uleft [IU];
  fright[ID] = uright[IU];
  fleft [IP] = qleft [IU] * ( uleft [IP] + qleft [IP]);
  fright[IP] = qright[IU] * ( uright[IP] + qright[IP]);
  fleft [IU] = qleft [IP] +   uleft [IU] * qleft [IU];
  fright[IU] = qright[IP] +   uright[IU] * qright[IU];

  // Other advected quantities
  fleft [IV] = fleft [ID] * qleft [IV];
  fright[IV] = fright[ID] * qright[IV];
  if (NVAR == NVAR_3D) {
    fleft [IW] = fleft [ID] * qleft [IW];
    fright[IW] = fright[ID] * qright[IW];
  }

  // Compute HLL fluxes
  for (int nVar=0; nVar < NVAR_2D; nVar++) {
    flux[nVar] = (SR * fleft[nVar] - SL * fright[nVar] + 
		  SR * SL * (uright[nVar] - uleft[nVar]) ) / (SR-SL);
  }
  if (NVAR == NVAR_3D) {
    flux[IW] = (SR * fleft[IW] - SL * fright[IW] + 
		SR * SL * (uright[IW] - uleft[IW]) ) / (SR-SL);
  }

} // riemann_hll


/** 
 * Riemann solver, equivalent to riemann_hllc in RAMSES (see file
 * godunov_utils.f90 in RAMSES).
 * 
 * Hydro ONLY. When doing MHD use hlld (see file riemann_mhd.h) !
 *
 * @param[in] qleft : input left state
 * @param[in] qright : input right state
 * @param[out] qgdnv : output Godunov state
 * @param[out] flux  : output flux
 */
template <NvarSimulation NVAR>
__DEVICE__
void riemann_hllc(real_t qleft[NVAR], real_t qright[NVAR], 
		  real_t (&qgdnv)[NVAR], real_t (&flux)[NVAR])
{

  (void) qgdnv;

  const real_t entho = ONE_F / (gParams.gamma0 - ONE_F);
  
  // Left variables
  real_t rl = FMAX(qleft[ID], gParams.smallr);
  real_t pl = FMAX(qleft[IP], rl*gParams.smallp);
  real_t ul =      qleft[IU];
    
  real_t ecinl = HALF_F*rl*ul*ul;
  ecinl += HALF_F*rl*qleft[IV]*qleft[IV];
  if (NVAR == NVAR_3D)
    ecinl += HALF_F*rl*qleft[IW]*qleft[IW];

  real_t  etotl = pl*entho+ecinl;
  real_t& ptotl = pl;

  // Right variables
  real_t rr = FMAX(qright[ID], gParams.smallr);
  real_t pr = FMAX(qright[IP], rr*gParams.smallp);
  real_t ur =      qright[IU];

  real_t ecinr = HALF_F*rr*ur*ur;
  ecinr += HALF_F*rr*qright[IV]*qright[IV];
  if (NVAR == NVAR_3D)
    ecinr += HALF_F*rr*qright[IW]*qright[IW];
  
  real_t  etotr = pr*entho+ecinr;
  real_t& ptotr = pr;
    
  // Find the largest eigenvalues in the normal direction to the interface
  real_t cfastl = SQRT(FMAX(gParams.gamma0*pl/rl,gParams.smallc*gParams.smallc));
  real_t cfastr = SQRT(FMAX(gParams.gamma0*pr/rr,gParams.smallc*gParams.smallc));

  // Compute HLL wave speed
  real_t SL = FMIN(ul,ur) - FMAX(cfastl,cfastr);
  real_t SR = FMAX(ul,ur) + FMAX(cfastl,cfastr);

  // Compute lagrangian sound speed
  real_t rcl = rl*(ul-SL);
  real_t rcr = rr*(SR-ur);
    
  // Compute acoustic star state
  real_t ustar    = (rcr*ur   +rcl*ul   +  (ptotl-ptotr))/(rcr+rcl);
  real_t ptotstar = (rcr*ptotl+rcl*ptotr+rcl*rcr*(ul-ur))/(rcr+rcl);

  // Left star region variables
  real_t rstarl    = rl*(SL-ul)/(SL-ustar);
  real_t etotstarl = ((SL-ul)*etotl-ptotl*ul+ptotstar*ustar)/(SL-ustar);
    
  // Right star region variables
  real_t rstarr    = rr*(SR-ur)/(SR-ustar);
  real_t etotstarr = ((SR-ur)*etotr-ptotr*ur+ptotstar*ustar)/(SR-ustar);
    
  // Sample the solution at x/t=0
  real_t ro, uo, ptoto, etoto;
  if (SL > ZERO_F) {
    ro=rl;
    uo=ul;
    ptoto=ptotl;
    etoto=etotl;
  } else if (ustar > ZERO_F) {
    ro=rstarl;
    uo=ustar;
    ptoto=ptotstar;
    etoto=etotstarl;
  } else if (SR > ZERO_F) {
    ro=rstarr;
    uo=ustar;
    ptoto=ptotstar;
    etoto=etotstarr;
  } else {
    ro=rr;
    uo=ur;
    ptoto=ptotr;
    etoto=etotr;
  }
      
  // Compute the Godunov flux
  flux[ID] = ro*uo;
  flux[IU] = ro*uo*uo+ptoto;
  flux[IP] = (etoto+ptoto)*uo;
  if (flux[ID] > ZERO_F) {
    flux[IV] = flux[ID]*qleft[IV];
  } else {
    flux[IV] = flux[ID]*qright[IV];
  }
  
  if (NVAR == NVAR_3D) {
    if (flux[ID] > ZERO_F) {
      flux[IW] = flux[ID]*qleft[IW];
    } else {
      flux[IW] = flux[ID]*qright[IW];
    }
  }

} // riemann_hllc




/** 
 * Wrapper routine to call the right Riemann solver routine according
 * to the global parameter riemannSolver
 * 
 * @param[in] qleft : input left state
 * @param[in] qright : input right state
 * @param[out] qgdnv : output Godunov state
 * @param[out] flux  : output flux vector
 *
 * template parameter:
 * @tparam NVAR size of state vector (can only be NVAR_2D, NVAR_3D or NVAR_MHD)
 */
template <NvarSimulation NVAR>
__DEVICE__
void riemann(real_t qleft[NVAR], real_t qright[NVAR], 
	     real_t (&qgdnv)[NVAR], real_t (&flux)[NVAR])
{
  if (gParams.riemannSolver == APPROX) {
    riemann_approx<NVAR>(qleft, qright, qgdnv, flux);
  } else if (gParams.riemannSolver == HLL) {
    riemann_hll<NVAR>(qleft,qright, qgdnv, flux);
  } else if (gParams.riemannSolver == HLLC) {
    riemann_hllc<NVAR>(qleft,qright, qgdnv, flux);
  }

} // riemann

#endif /*RIEMANN_H_*/
