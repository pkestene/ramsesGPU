/**
 * \file riemann_mhd.h
 * \brief Provides CPU/GPU riemann solver routines for MHD.
 *
 * We can't put everything in a single header because it might lead to
 * linking problems if the file is included in more than one
 * implementation file.
 *
 * It is also annoying to put these routine in a dedicated
 * implementation (.cpp / .cu) file since it requires the global
 * parameter object to be initialized (make a call to
 * copyToSymbolMemory to have those parameter in the GPU constant
 * memory space).
 *
 * \date March 31, 2011
 * \author Pierre Kestener
 *
 * $Id: riemann_mhd.h 3450 2014-06-16 22:03:23Z pkestene $
 */
#ifndef RIEMANN_MHD_H_
#define RIEMANN_MHD_H_

#include "real_type.h"
#include "constants.h"

// some MHD utilities
#include "mhd_utils.h"

/*
 * MHD HLL Riemann solver
 *
 * qleft, qright and flux have now NVAR_MHD=8 components.
 *
 * The following code is adapted from Dumses.
 *
 * @param[in] qleft  : input left state
 * @param[in] qright : input right state
 * @param[out] flux  : output flux
 *
 */
__DEVICE__ inline
void riemann_hll(real_riemann_t   qleft[NVAR_MHD],
		 real_riemann_t  qright[NVAR_MHD], 
		 real_riemann_t (&flux)[NVAR_MHD])
{
  
  // enforce continuity of normal component
  real_riemann_t bx_mean = HALF_F * ( qleft[IA] + qright[IA] );
  qleft [IA] = bx_mean;
  qright[IA] = bx_mean;
  
  real_riemann_t uleft[NVAR_MHD],  fleft[NVAR_MHD];
  real_riemann_t uright[NVAR_MHD], fright[NVAR_MHD];
  
  find_mhd_flux(qleft ,uleft ,fleft );
  find_mhd_flux(qright,uright,fright);
  
  // find the largest eigenvalue in the normal direction to the interface
  real_riemann_t cfleft  = find_speed_fast<IX>(qleft);
  real_riemann_t cfright = find_speed_fast<IX>(qright);
  
  real_riemann_t vleft =qleft [IU];
  real_riemann_t vright=qright[IU];
  real_riemann_t sl=FMIN ( FMIN (vleft,vright) - FMAX (cfleft,cfright) , ZERO_F);
  real_riemann_t sr=FMAX ( FMAX (vleft,vright) + FMAX (cfleft,cfright) , ZERO_F);
  
  // the hll flux
  for (int iVar=0; iVar<NVAR_MHD; iVar++)
    flux[iVar] = (sr*fleft[iVar]-sl*fright[iVar]+sr*sl*(uright[iVar]-uleft[iVar]))/(sr-sl);
  
} // riemann_hll

/*
 * MHD LLF (Local Lax-Friedrich) Riemann solver
 *
 * qleft, qright and flux have now NVAR_MHD=8 components.
 *
 * The following code is adapted from Dumses.
 *
 * @param[in] qleft  : input left state
 * @param[in] qright : input right state
 * @param[out] flux  : output flux
 * @param[in] zero_flux : when riemann_llf is used to compute EMF, zero_flux should be ZERO_F
 *
 */
__DEVICE__ inline
void riemann_llf(real_riemann_t   qleft[NVAR_MHD], 
		 real_riemann_t  qright[NVAR_MHD], 
		 real_riemann_t (&flux)[NVAR_MHD],
		 real_riemann_t  zero_flux=ONE_F)
{
  
  // enforce continuity of normal component
  real_riemann_t bx_mean = HALF_F * ( qleft[IA] + qright[IA] );
  qleft [IA] = bx_mean;
  qright[IA] = bx_mean;
  
  real_riemann_t uleft[NVAR_MHD],  fleft[NVAR_MHD];
  real_riemann_t uright[NVAR_MHD], fright[NVAR_MHD];
  
  find_mhd_flux(qleft ,uleft ,fleft );
  find_mhd_flux(qright,uright,fright);
  
  // compute mean flux
  for (int iVar=0; iVar<NVAR_MHD; iVar++)
    flux[iVar] = (qleft[iVar]+qright[iVar])/2*zero_flux;

  // find the largest eigenvalue in the normal direction to the interface
  real_riemann_t cleft  = find_speed_info(qleft);
  real_riemann_t cright = find_speed_info(qright);
  
  real_riemann_t vel_info = FMAX(cleft,cright);

  // the Local Lax-Friedrich flux
  for (int iVar=0; iVar<NVAR_MHD; iVar++)
    flux[iVar] -= vel_info*(uright[iVar]-uleft[iVar])/2;
  
} // riemann_llf

/** 
 * Riemann solver, equivalent to riemann_hlld in RAMSES/DUMSES (see file
 * godunov_utils.f90 in RAMSES/DUMSES).
 *
 * Reference :
 * <A HREF="http://www.sciencedirect.com/science/article/B6WHY-4FY3P80-7/2/426234268c96dcca8a828d098b75fe4e">
 * Miyoshi & Kusano, 2005, JCP, 208, 315 </A>
 *
 * \warning This version of HLLD integrates the pressure term in
 * flux[IU] (as in RAMSES). This will need to be modified in the
 * future (as it is done in DUMSES) to handle cylindrical / spherical
 * coordinate systems. For example, one could add a new ouput named qStar
 * to store star state, and that could be used to compute geometrical terms
 * outside this routine.
 *
 * @param[in] qleft : input left state
 * @param[in] qright : input right state
 * @param[out] flux  : output flux
 */
__DEVICE__ inline
void riemann_hlld(real_riemann_t   qleft[NVAR_MHD],
		  real_riemann_t  qright[NVAR_MHD], 
		  real_riemann_t (&flux)[NVAR_MHD])
{

  // Constants
  const real_riemann_t entho = ONE_F / (gParams.gamma0 - ONE_F);

  // Enforce continuity of normal component of magnetic field
  real_riemann_t a    = HALF_F * ( qleft[IA] + qright[IA] );
  real_riemann_t sgnm = (a >= 0) ? ONE_F : -ONE_F;
  qleft [IA]  = a; 
  qright[IA]  = a;
  
  // ISOTHERMAL
  real_t &cIso = ::gParams.cIso;
  if (cIso > 0) {
    // recompute pressure
    qleft [IP] = qleft [ID]*cIso*cIso;
    qright[IP] = qright[ID]*cIso*cIso;
  } // end ISOTHERMAL

  // left variables
  real_riemann_t rl, pl, ul, vl, wl, bl, cl;
  rl = qleft[ID]; //rl = FMAX(qleft[ID], static_cast<real_riemann_t>(gParams.smallr)    );  
  pl = qleft[IP]; //pl = FMAX(qleft[IP], static_cast<real_riemann_t>(rl*gParams.smallp) ); 
  ul = qleft[IU];  vl = qleft[IV];  wl = qleft[IW]; 
  bl = qleft[IB];  cl = qleft[IC];
  real_riemann_t ecinl = HALF_F * (ul*ul + vl*vl + wl*wl) * rl;
  real_riemann_t emagl = HALF_F * ( a*a  + bl*bl + cl*cl);
  real_riemann_t etotl = pl*entho + ecinl + emagl;
  real_riemann_t ptotl = pl + emagl;
  real_riemann_t vdotbl= ul*a + vl*bl + wl*cl;

  // right variables
  real_riemann_t rr, pr, ur, vr, wr, br, cr;
  rr = qright[ID]; //rr = FMAX(qright[ID], static_cast<real_riemann_t>( gParams.smallr) );
  pr = qright[IP]; //pr = FMAX(qright[IP], static_cast<real_riemann_t>( rr*gParams.smallp) ); 
  ur = qright[IU];  vr=qright[IV];  wr = qright[IW]; 
  br = qright[IB];  cr=qright[IC];
  real_riemann_t ecinr = HALF_F * (ur*ur + vr*vr + wr*wr) * rr;
  real_riemann_t emagr = HALF_F * ( a*a  + br*br + cr*cr);
  real_riemann_t etotr = pr*entho + ecinr + emagr;
  real_riemann_t ptotr = pr + emagr;
  real_riemann_t vdotbr= ur*a + vr*br + wr*cr;

  // find the largest eigenvalues in the normal direction to the interface
  real_riemann_t cfastl = find_speed_fast<IX>(qleft);
  real_riemann_t cfastr = find_speed_fast<IX>(qright);

  // compute hll wave speed
  real_riemann_t sl = FMIN(ul,ur) - FMAX(cfastl,cfastr);
  real_riemann_t sr = FMAX(ul,ur) + FMAX(cfastl,cfastr);
  
  // compute lagrangian sound speed
  real_riemann_t rcl = rl * (ul-sl);
  real_riemann_t rcr = rr * (sr-ur);

  // compute acoustic star state
  real_riemann_t ustar   = (rcr*ur   +rcl*ul   +  (ptotl-ptotr))/(rcr+rcl);
  real_riemann_t ptotstar= (rcr*ptotl+rcl*ptotr+rcl*rcr*(ul-ur))/(rcr+rcl);

  // left star region variables
  real_riemann_t estar;
  real_riemann_t rstarl, el;
  rstarl = rl*(sl-ul)/(sl-ustar);
  estar  = rl*(sl-ul)*(sl-ustar)-a*a;
  el     = rl*(sl-ul)*(sl-ul   )-a*a;
  real_riemann_t vstarl, wstarl;
  real_riemann_t bstarl, cstarl;
  // not very good (should use a small energy cut-off !!!)
  if(a*a>0 and FABS(estar/(a*a)-ONE_F)<=1e-8) {
    vstarl=vl;
    bstarl=bl;
    wstarl=wl;
    cstarl=cl;
  } else {
    vstarl=vl-a*bl*(ustar-ul)/estar;
    bstarl=bl*el/estar;
    wstarl=wl-a*cl*(ustar-ul)/estar;
    cstarl=cl*el/estar;
  }
  real_riemann_t vdotbstarl = ustar*a+vstarl*bstarl+wstarl*cstarl;
  real_riemann_t etotstarl  = ((sl-ul)*etotl-ptotl*ul+ptotstar*ustar+a*(vdotbl-vdotbstarl))/(sl-ustar);
  real_riemann_t sqrrstarl  = SQRT(rstarl);
  real_riemann_t calfvenl   = FABS(a)/sqrrstarl; /* sqrrstarl should never be zero, but it might happen if border conditions are not OK !!!!!! */
  real_riemann_t sal        = ustar-calfvenl;

  // right star region variables
  real_riemann_t rstarr, er;
  rstarr = rr*(sr-ur)/(sr-ustar);
  estar  = rr*(sr-ur)*(sr-ustar)-a*a;
  er     = rr*(sr-ur)*(sr-ur   )-a*a;
  real_riemann_t vstarr, wstarr;
  real_riemann_t bstarr, cstarr;
  // not very good (should use a small energy cut-off !!!)
  if(a*a>0 and FABS(estar/(a*a)-ONE_F)<=1e-8) {
    vstarr=vr;
    bstarr=br;
    wstarr=wr;
    cstarr=cr;
  } else {
    vstarr=vr-a*br*(ustar-ur)/estar;
    bstarr=br*er/estar;
    wstarr=wr-a*cr*(ustar-ur)/estar;
    cstarr=cr*er/estar;
  }
  real_riemann_t vdotbstarr = ustar*a+vstarr*bstarr+wstarr*cstarr;
  real_riemann_t etotstarr  = ((sr-ur)*etotr-ptotr*ur+ptotstar*ustar+a*(vdotbr-vdotbstarr))/(sr-ustar);
  real_riemann_t sqrrstarr  = SQRT(rstarr);
  real_riemann_t calfvenr   = FABS(a)/sqrrstarr; /* sqrrstarr should never be zero, but it might happen if border conditions are not OK !!!!!! */
  real_riemann_t sar        = ustar+calfvenr;

  // double star region variables
  real_riemann_t vstarstar     = (sqrrstarl*vstarl+sqrrstarr*vstarr+
			 sgnm*(bstarr-bstarl)) / (sqrrstarl+sqrrstarr);
  real_riemann_t wstarstar     = (sqrrstarl*wstarl+sqrrstarr*wstarr+
			 sgnm*(cstarr-cstarl)) / (sqrrstarl+sqrrstarr);
  real_riemann_t bstarstar     = (sqrrstarl*bstarr+sqrrstarr*bstarl+
			 sgnm*sqrrstarl*sqrrstarr*(vstarr-vstarl)) / 
    (sqrrstarl+sqrrstarr);
  real_riemann_t cstarstar     = (sqrrstarl*cstarr+sqrrstarr*cstarl+
			 sgnm*sqrrstarl*sqrrstarr*(wstarr-wstarl)) /
    (sqrrstarl+sqrrstarr);
  real_riemann_t vdotbstarstar = ustar*a+vstarstar*bstarstar+wstarstar*cstarstar;
  real_riemann_t etotstarstarl = etotstarl-sgnm*sqrrstarl*(vdotbstarl-vdotbstarstar);
  real_riemann_t etotstarstarr = etotstarr+sgnm*sqrrstarr*(vdotbstarr-vdotbstarstar);
  
  // sample the solution at x/t=0
  real_riemann_t ro, uo, vo, wo, bo, co, ptoto, etoto, vdotbo;
  if(sl>0) { // flow is supersonic, return upwind variables
    ro=rl;
    uo=ul;
    vo=vl;
    wo=wl;
    bo=bl;
    co=cl;
    ptoto=ptotl;
    etoto=etotl;
    vdotbo=vdotbl;
  } else if (sal>0) {
    ro=rstarl;
    uo=ustar;
    vo=vstarl;
    wo=wstarl;
    bo=bstarl;
    co=cstarl;
    ptoto=ptotstar;
    etoto=etotstarl;
    vdotbo=vdotbstarl;
  } else if (ustar>0) {
    ro=rstarl;
    uo=ustar;
    vo=vstarstar;
    wo=wstarstar;
    bo=bstarstar;
    co=cstarstar;
    ptoto=ptotstar;
    etoto=etotstarstarl;
    vdotbo=vdotbstarstar;
  } else if (sar>0) {
    ro=rstarr;
    uo=ustar;
    vo=vstarstar;
    wo=wstarstar;
    bo=bstarstar;
    co=cstarstar;
    ptoto=ptotstar;
    etoto=etotstarstarr;
    vdotbo=vdotbstarstar;
  } else if (sr>0) {
    ro=rstarr;
    uo=ustar;
    vo=vstarr;
    wo=wstarr;
    bo=bstarr;
    co=cstarr;
    ptoto=ptotstar;
    etoto=etotstarr;
    vdotbo=vdotbstarr;
  } else { // flow is supersonic, return upwind variables
    ro=rr;
    uo=ur;
    vo=vr;
    wo=wr;
    bo=br;
    co=cr;
    ptoto=ptotr;
    etoto=etotr;
    vdotbo=vdotbr;
  }

  // compute the godunov flux
  flux[ID] = ro*uo;
  flux[IP] = (etoto+ptoto)*uo-a*vdotbo;
  flux[IU] = ro*uo*uo-a*a+ptoto; /* *** WARNING *** : ptoto used here (this is only valid for cartesian geometry) ! */
  flux[IV] = ro*uo*vo-a*bo;
  flux[IW] = ro*uo*wo-a*co;
  flux[IA] = ZERO_F;
  flux[IB] = bo*uo-a*vo;
  flux[IC] = co*uo-a*wo;
  
} // riemann_hlld

/** 
 * Wrapper routine to call the right Riemann solver routine according
 * to the global parameter riemannSolver
 * 
 * @param[in] qleft : input left state
 * @param[in] qright : input right state
 * @param[out] flux  : output flux vector
 *
 * @TODO add a fourth argument to store starState
 */
__DEVICE__ inline
void riemann_mhd(real_riemann_t   qleft[NVAR_MHD],
		 real_riemann_t  qright[NVAR_MHD], 
		 real_riemann_t (&flux)[NVAR_MHD])
{
  
  if (gParams.riemannSolver == HLL) {
    riemann_hll(qleft,qright, flux);
  } else if (gParams.riemannSolver == LLF) {
    riemann_llf(qleft,qright, flux);
  } else if (gParams.riemannSolver == HLLD) {
    riemann_hlld(qleft,qright, flux);
  }
  
} // riemann_mhd

/**
 * max value out of 4
 */
__DEVICE__ inline
real_t FMAX4(real_t a0, real_t a1, real_t a2, real_t a3)
{
  real_t returnVal = a0;
  returnVal = ( a1 > returnVal) ? a1 : returnVal;
  returnVal = ( a2 > returnVal) ? a2 : returnVal;
  returnVal = ( a3 > returnVal) ? a3 : returnVal;

  return returnVal;
} // FMAX4

/**
 * min value out of 4
 */
__DEVICE__ inline
real_t FMIN4(real_t a0, real_t a1, real_t a2, real_t a3)
{
  real_t returnVal = a0;
  returnVal = ( a1 < returnVal) ? a1 : returnVal;
  returnVal = ( a2 < returnVal) ? a2 : returnVal;
  returnVal = ( a3 < returnVal) ? a3 : returnVal;

  return returnVal;
} // FMIN4

/**
 * max value out of 5
 */
__DEVICE__ inline
real_t FMAX5(real_t a0, real_t a1, real_t a2, real_t a3, real_t a4)
{
  real_t returnVal = a0;
  returnVal = ( a1 > returnVal) ? a1 : returnVal;
  returnVal = ( a2 > returnVal) ? a2 : returnVal;
  returnVal = ( a3 > returnVal) ? a3 : returnVal;
  returnVal = ( a4 > returnVal) ? a4 : returnVal;

  return returnVal;
} // FMAX5

/**
 * 2D magnetic riemann solver of type HLLA
 *
 */
__DEVICE__ inline
real_t mag_riemann2d_hlla(real_t qLLRR[4][NVAR_MHD],
			  real_t eLLRR[4])
{

  // alias reference to input arrays
  real_t (&qLL)[NVAR_MHD] = qLLRR[ILL];
  real_t (&qRL)[NVAR_MHD] = qLLRR[IRL];
  real_t (&qLR)[NVAR_MHD] = qLLRR[ILR];
  real_t (&qRR)[NVAR_MHD] = qLLRR[IRR];

  // Compute 4 Alfven velocity relative to x direction
  real_t &vLLx = qLL[IU]; real_t cLLx = find_speed_alfven(qLL[ID], qLL[IA]);
  real_t &vLRx = qLR[IU]; real_t cLRx = find_speed_alfven(qLR[ID], qLR[IA]);
  real_t &vRLx = qRL[IU]; real_t cRLx = find_speed_alfven(qRL[ID], qRL[IA]);
  real_t &vRRx = qRR[IU]; real_t cRRx = find_speed_alfven(qRR[ID], qRR[IA]);
  real_t cMaxx = FMAX5(cLLx,cLRx,cRLx,cRRx,gParams.smallc);

  real_t &vLLy = qLL[IV]; real_t cLLy = find_speed_alfven(qLL[ID], qLL[IB]);
  real_t &vLRy = qLR[IV]; real_t cLRy = find_speed_alfven(qLR[ID], qLR[IB]);
  real_t &vRLy = qRL[IV]; real_t cRLy = find_speed_alfven(qRL[ID], qRL[IB]);
  real_t &vRRy = qRR[IV]; real_t cRRy = find_speed_alfven(qRR[ID], qRR[IB]);
  real_t cMaxy = FMAX5(cLLy,cLRy,cRLy,cRRy,gParams.smallc);

  real_t SL, SR, SB, ST;
  SL = FMIN(FMIN4(vLLx,vLRx,vRLx,vRRx)-cMaxx,ZERO_F);
  SR = FMAX(FMAX4(vLLx,vLRx,vRLx,vRRx)+cMaxx,ZERO_F);
  SB = FMIN(FMIN4(vLLy,vLRy,vRLy,vRRy)-cMaxy,ZERO_F);
  ST = FMAX(FMAX4(vLLy,vLRy,vRLy,vRRy)+cMaxy,ZERO_F);

  real_t  E   = 0;
  real_t &ELL = eLLRR[ILL];
  real_t &ERL = eLLRR[IRL];
  real_t &ELR = eLLRR[ILR];
  real_t &ERR = eLLRR[IRR];

  E = (SL*SB*ERR-SL*ST*ERL-SR*SB*ELR+SR*ST*ELL)/(SR-SL)/(ST-SB)
    - ST*SB/(ST-SB)*(qRR[IA]-qLL[IA])
    + SR*SL/(SR-SL)*(qRR[IB]-qLL[IB]);

  return E;

} // mag_riemann_hlla

/**
 * 2D magnetic riemann solver of type HLLF
 *
 */
__DEVICE__ inline
real_t mag_riemann2d_hllf(real_t qLLRR[4][NVAR_MHD],
			  real_t eLLRR[4])
{

  // alias reference to input arrays
  real_t (&qLL)[NVAR_MHD] = qLLRR[ILL];
  real_t (&qRL)[NVAR_MHD] = qLLRR[IRL];
  real_t (&qLR)[NVAR_MHD] = qLLRR[ILR];
  real_t (&qRR)[NVAR_MHD] = qLLRR[IRR];

  // Compute 4 Alfven velocity relative to x direction
  real_t &vLLx = qLL[IU]; real_t cLLx = find_speed_fast<IX>(qLL);
  real_t &vLRx = qLR[IU]; real_t cLRx = find_speed_fast<IX>(qLR);
  real_t &vRLx = qRL[IU]; real_t cRLx = find_speed_fast<IX>(qRL);
  real_t &vRRx = qRR[IU]; real_t cRRx = find_speed_fast<IX>(qRR);
  real_t cMaxx = FMAX4(cLLx,cLRx,cRLx,cRRx);

  real_t &vLLy = qLL[IV]; real_t cLLy = find_speed_fast<IY>(qLL);
  real_t &vLRy = qLR[IV]; real_t cLRy = find_speed_fast<IY>(qLR);
  real_t &vRLy = qRL[IV]; real_t cRLy = find_speed_fast<IY>(qRL);
  real_t &vRRy = qRR[IV]; real_t cRRy = find_speed_fast<IY>(qRR);
  real_t cMaxy = FMAX4(cLLy,cLRy,cRLy,cRRy);

  real_t SL, SR, SB, ST;
  SL = FMIN(FMIN4(vLLx,vLRx,vRLx,vRRx)-cMaxx,ZERO_F);
  SR = FMAX(FMAX4(vLLx,vLRx,vRLx,vRRx)+cMaxx,ZERO_F);
  SB = FMIN(FMIN4(vLLy,vLRy,vRLy,vRRy)-cMaxy,ZERO_F);
  ST = FMAX(FMAX4(vLLy,vLRy,vRLy,vRRy)+cMaxy,ZERO_F);

  real_t  E   = 0;
  real_t &ELL = eLLRR[ILL];
  real_t &ERL = eLLRR[IRL];
  real_t &ELR = eLLRR[ILR];
  real_t &ERR = eLLRR[IRR];

  E = (SL*SB*ERR-SL*ST*ERL-SR*SB*ELR+SR*ST*ELL)/(SR-SL)/(ST-SB)
    - ST*SB/(ST-SB)*(qRR[IA]-qLL[IA])
    + SR*SL/(SR-SL)*(qRR[IB]-qLL[IB]);

  return E;

} // mag_riemann_hllf

/**
 * 2D magnetic riemann solver of type LLF (Local Lax-Friedrich)
 *
 * \see compute_emf Here compoent have been swapped so that
 * iu, iv : parallel velocity indexes
 * iw     : orthogonal velocity index
 * ia, ib, ic : idem for magnetic field
 * That the way ramses/Dumses is !
 */
__DEVICE__ inline
real_t mag_riemann2d_llf(real_t qLLRR[4][NVAR_MHD],
			 real_t eLLRR[4])
{

  // alias reference to input arrays
  real_t (&qLL)[NVAR_MHD] = qLLRR[ILL];
  real_t (&qRL)[NVAR_MHD] = qLLRR[IRL];
  real_t (&qLR)[NVAR_MHD] = qLLRR[ILR];
  real_t (&qRR)[NVAR_MHD] = qLLRR[IRR];

  real_t &ELL = eLLRR[ILL];
  real_t &ERL = eLLRR[IRL];
  real_t &ELR = eLLRR[ILR];
  real_t &ERR = eLLRR[IRR];

  real_t E = (ELL+ERL+ELR+ERR)/4;

  real_riemann_t qleft[NVAR_MHD];
  real_riemann_t qright[NVAR_MHD];

  const real_riemann_t zero_flux = ZERO_F;
  
  /*
   * call the first solver in the x direction 
   */
  //density
  qleft[ID]  = (qLL[ID]+qLR[ID])/2;
  qright[ID] = (qRR[ID]+qRL[ID])/2;

  // pressure
  qleft [IP] = (qLL[IP]+qLR[IP])/2;
  qright[IP] = (qRR[IP]+qRL[IP])/2;

  qleft [IU] = (qLL[IU]+qLR[IU])/2;
  qright[IU] = (qRR[IU]+qRL[IU])/2;
  
  qleft [IV] = (qLL[IV]+qLR[IV])/2;
  qright[IV] = (qRR[IV]+qRL[IV])/2;
  
  qleft [IW] = (qLL[IW]+qLR[IW])/2;
  qright[IW] = (qRR[IW]+qRL[IW])/2;
  
  qleft [IA] = (qLL[IA]+qLR[IA])/2;
  qright[IA] = (qRR[IA]+qRL[IA])/2;
  
  qleft [IB] = (qLL[IB]+qLR[IB])/2;
  qright[IB] = (qRR[IB]+qRL[IB])/2;
  
  qleft [IC] = (qLL[IC]+qLR[IC])/2;
  qright[IC] = (qRR[IC]+qRL[IC])/2;
  
  real_riemann_t fmean_x[NVAR_MHD];
  riemann_llf(qleft,qright,fmean_x,zero_flux);

  /*
   * call the second solver in the y direction
   */
  // density
  qleft [ID] = (qLL[ID]+qRL[ID])/2;
  qright[ID] = (qRR[ID]+qLR[ID])/2;
  
  // pressure
  qleft [IP] = (qLL[IP]+qRL[IP])/2;
  qright[IP] = (qRR[IP]+qLR[IP])/2;
  
  qleft [IU] = (qLL[IV]+qRL[IV])/2;
  qright[IU] = (qRR[IV]+qLR[IV])/2;

  qleft [IV] = (qLL[IU]+qRL[IU])/2;
  qright[IV] = (qRR[IU]+qLR[IU])/2;
  
  qleft [IW] = (qLL[IW]+qRL[IW])/2;
  qright[IW] = (qRR[IW]+qLR[IW])/2;
  
  qleft [IA] = (qLL[IB]+qRL[IB])/2;
  qright[IA] = (qRR[IB]+qLR[IB])/2;
    
  qleft [IB] = (qLL[IA]+qRL[IA])/2;
  qright[IB] = (qRR[IA]+qLR[IA])/2;

  qleft [IC] = (qLL[IC]+qRL[IC])/2;
  qright[IC] = (qRR[IC]+qLR[IC])/2;

  real_riemann_t fmean_y[NVAR_MHD];
  riemann_llf(qleft,qright,fmean_y,zero_flux);

  E += (fmean_x[IB] - fmean_y[IB]);

  return E;
  
} // mag_riemann2d_llf

/**
 * 2D magnetic riemann solver of type HLLD
 *
 */
__DEVICE__ inline
real_t mag_riemann2d_hlld(real_t qLLRR[4][NVAR_MHD],
			  real_t eLLRR[4])
{

  // alias reference to input arrays
  real_t (&qLL)[NVAR_MHD] = qLLRR[ILL];
  real_t (&qRL)[NVAR_MHD] = qLLRR[IRL];
  real_t (&qLR)[NVAR_MHD] = qLLRR[ILR];
  real_t (&qRR)[NVAR_MHD] = qLLRR[IRR];

  real_t &ELL = eLLRR[ILL];
  real_t &ERL = eLLRR[IRL];
  real_t &ELR = eLLRR[ILR];
  real_t &ERR = eLLRR[IRR];
  //real_t ELL,ERL,ELR,ERR;

  real_t &rLL=qLL[ID]; real_t &pLL=qLL[IP]; 
  real_t &uLL=qLL[IU]; real_t &vLL=qLL[IV]; 
  real_t &aLL=qLL[IA]; real_t &bLL=qLL[IB] ; real_t &cLL=qLL[IC];
  
  real_t &rLR=qLR[ID]; real_t &pLR=qLR[IP]; 
  real_t &uLR=qLR[IU]; real_t &vLR=qLR[IV]; 
  real_t &aLR=qLR[IA]; real_t &bLR=qLR[IB] ; real_t &cLR=qLR[IC];
  
  real_t &rRL=qRL[ID]; real_t &pRL=qRL[IP]; 
  real_t &uRL=qRL[IU]; real_t &vRL=qRL[IV]; 
  real_t &aRL=qRL[IA]; real_t &bRL=qRL[IB] ; real_t &cRL=qRL[IC];

  real_t &rRR=qRR[ID]; real_t &pRR=qRR[IP]; 
  real_t &uRR=qRR[IU]; real_t &vRR=qRR[IV]; 
  real_t &aRR=qRR[IA]; real_t &bRR=qRR[IB] ; real_t &cRR=qRR[IC];
  
  // Compute 4 fast magnetosonic velocity relative to x direction
  real_t cFastLLx = find_speed_fast<IX>(qLL);
  real_t cFastLRx = find_speed_fast<IX>(qLR);
  real_t cFastRLx = find_speed_fast<IX>(qRL);
  real_t cFastRRx = find_speed_fast<IX>(qRR);

  // Compute 4 fast magnetosonic velocity relative to y direction 
  real_t cFastLLy = find_speed_fast<IY>(qLL);
  real_t cFastLRy = find_speed_fast<IY>(qLR);
  real_t cFastRLy = find_speed_fast<IY>(qRL);
  real_t cFastRRy = find_speed_fast<IY>(qRR);
  
  // TODO : write a find_speed that computes the 2 speeds together (in
  // a single routine -> factorize computation of cFastLLx and cFastLLy

  real_t SL = FMIN4(uLL,uLR,uRL,uRR) - FMAX4(cFastLLx,cFastLRx,cFastRLx,cFastRRx);
  real_t SR = FMAX4(uLL,uLR,uRL,uRR) + FMAX4(cFastLLx,cFastLRx,cFastRLx,cFastRRx);
  real_t SB = FMIN4(vLL,vLR,vRL,vRR) - FMAX4(cFastLLy,cFastLRy,cFastRLy,cFastRRy);
  real_t ST = FMAX4(vLL,vLR,vRL,vRR) + FMAX4(cFastLLy,cFastLRy,cFastRLy,cFastRRy);

  /*ELL = uLL*bLL - vLL*aLL;
    ELR = uLR*bLR - vLR*aLR;
    ERL = uRL*bRL - vRL*aRL;
    ERR = uRR*bRR - vRR*aRR;*/
  
  real_t PtotLL = pLL + HALF_F * (aLL*aLL + bLL*bLL + cLL*cLL);
  real_t PtotLR = pLR + HALF_F * (aLR*aLR + bLR*bLR + cLR*cLR);
  real_t PtotRL = pRL + HALF_F * (aRL*aRL + bRL*bRL + cRL*cRL);
  real_t PtotRR = pRR + HALF_F * (aRR*aRR + bRR*bRR + cRR*cRR);
  
  real_t rcLLx = rLL * (uLL-SL); real_t rcRLx = rRL *(SR-uRL);
  real_t rcLRx = rLR * (uLR-SL); real_t rcRRx = rRR *(SR-uRR);
  real_t rcLLy = rLL * (vLL-SB); real_t rcLRy = rLR *(ST-vLR);
  real_t rcRLy = rRL * (vRL-SB); real_t rcRRy = rRR *(ST-vRR);

  real_t ustar = (rcLLx*uLL + rcLRx*uLR + rcRLx*uRL + rcRRx*uRR +
		  (PtotLL - PtotRL + PtotLR - PtotRR) ) / (rcLLx + rcLRx + 
							   rcRLx + rcRRx);
  real_t vstar = (rcLLy*vLL + rcLRy*vLR + rcRLy*vRL + rcRRy*vRR +
		  (PtotLL - PtotLR + PtotRL - PtotRR) ) / (rcLLy + rcLRy + 
							   rcRLy + rcRRy);
  
  real_t rstarLLx = rLL * (SL-uLL) / (SL-ustar);
  real_t BstarLL  = bLL * (SL-uLL) / (SL-ustar);
  real_t rstarLLy = rLL * (SB-vLL) / (SB-vstar); 
  real_t AstarLL  = aLL * (SB-vLL) / (SB-vstar);
  real_t rstarLL  = rLL * (SL-uLL) / (SL-ustar) 
    *                     (SB-vLL) / (SB-vstar);
  real_t EstarLLx = ustar * BstarLL - vLL   * aLL;
  real_t EstarLLy = uLL   * bLL     - vstar * AstarLL;
  real_t EstarLL  = ustar * BstarLL - vstar * AstarLL;
  
  real_t rstarLRx = rLR * (SL-uLR) / (SL-ustar); 
  real_t BstarLR  = bLR * (SL-uLR) / (SL-ustar);
  real_t rstarLRy = rLR * (ST-vLR) / (ST-vstar); 
  real_t AstarLR  = aLR * (ST-vLR) / (ST-vstar);
  real_t rstarLR  = rLR * (SL-uLR) / (SL-ustar) * (ST-vLR) / (ST-vstar);
  real_t EstarLRx = ustar * BstarLR - vLR   * aLR;
  real_t EstarLRy = uLR   * bLR     - vstar * AstarLR;
  real_t EstarLR  = ustar * BstarLR - vstar * AstarLR;

  real_t rstarRLx = rRL * (SR-uRL) / (SR-ustar); 
  real_t BstarRL  = bRL * (SR-uRL) / (SR-ustar);
  real_t rstarRLy = rRL * (SB-vRL) / (SB-vstar); 
  real_t AstarRL  = aRL * (SB-vRL) / (SB-vstar);
  real_t rstarRL  = rRL * (SR-uRL) / (SR-ustar) * (SB-vRL) / (SB-vstar);
  real_t EstarRLx = ustar * BstarRL - vRL   * aRL;
  real_t EstarRLy = uRL   * bRL     - vstar * AstarRL;
  real_t EstarRL  = ustar * BstarRL - vstar * AstarRL;
  
  real_t rstarRRx = rRR * (SR-uRR) / (SR-ustar); 
  real_t BstarRR  = bRR * (SR-uRR) / (SR-ustar);
  real_t rstarRRy = rRR * (ST-vRR) / (ST-vstar); 
  real_t AstarRR  = aRR * (ST-vRR) / (ST-vstar);
  real_t rstarRR  = rRR * (SR-uRR) / (SR-ustar) * (ST-vRR) / (ST-vstar);
  real_t EstarRRx = ustar * BstarRR - vRR   * aRR;
  real_t EstarRRy = uRR   * bRR     - vstar * AstarRR;
  real_t EstarRR  = ustar * BstarRR - vstar * AstarRR;

  real_t calfvenL = FMAX5(FABS(aLR)/SQRT(rstarLRx), FABS(AstarLR)/SQRT(rstarLR), 
			  FABS(aLL)/SQRT(rstarLLx), FABS(AstarLL)/SQRT(rstarLL), 
			  gParams.smallc);
  real_t calfvenR = FMAX5(FABS(aRR)/SQRT(rstarRRx), FABS(AstarRR)/SQRT(rstarRR),
			  FABS(aRL)/SQRT(rstarRLx), FABS(AstarRL)/SQRT(rstarRL), 
			  gParams.smallc);
  real_t calfvenB = FMAX5(FABS(bLL)/SQRT(rstarLLy), FABS(BstarLL)/SQRT(rstarLL), 
			  FABS(bRL)/SQRT(rstarRLy), FABS(BstarRL)/SQRT(rstarRL), 
			  gParams.smallc);
  real_t calfvenT = FMAX5(FABS(bLR)/SQRT(rstarLRy), FABS(BstarLR)/SQRT(rstarLR), 
			  FABS(bRR)/SQRT(rstarRRy), FABS(BstarRR)/SQRT(rstarRR), 
			  gParams.smallc);

  real_t SAL = FMIN(ustar - calfvenL, (real_t) ZERO_F); 
  real_t SAR = FMAX(ustar + calfvenR, (real_t) ZERO_F);
  real_t SAB = FMIN(vstar - calfvenB, (real_t) ZERO_F); 
  real_t SAT = FMAX(vstar + calfvenT, (real_t) ZERO_F);

  real_t AstarT = (SAR*AstarRR - SAL*AstarLR) / (SAR-SAL); 
  real_t AstarB = (SAR*AstarRL - SAL*AstarLL) / (SAR-SAL);
  
  real_t BstarR = (SAT*BstarRR - SAB*BstarRL) / (SAT-SAB); 
  real_t BstarL = (SAT*BstarLR - SAB*BstarLL) / (SAT-SAB);

  // finally get emf E
  real_t E=0, tmpE=0;

  // the following part is slightly different from the original fortran
  // code since it has to much different branches
  // which generate to much branch divergence in CUDA !!!

  // compute sort of boolean (don't know if signbit is available)
  int SB_pos = (int) (1+COPYSIGN(ONE_F,SB))/2, SB_neg = 1-SB_pos;
  int ST_pos = (int) (1+COPYSIGN(ONE_F,ST))/2, ST_neg = 1-ST_pos;
  int SL_pos = (int) (1+COPYSIGN(ONE_F,SL))/2, SL_neg = 1-SL_pos;
  int SR_pos = (int) (1+COPYSIGN(ONE_F,SR))/2, SR_neg = 1-SR_pos;

  // else
  tmpE = (SAL*SAB*EstarRR-SAL*SAT*EstarRL - 
	  SAR*SAB*EstarLR+SAR*SAT*EstarLL)/(SAR-SAL)/(SAT-SAB) - 
    SAT*SAB/(SAT-SAB)*(AstarT-AstarB) + 
    SAR*SAL/(SAR-SAL)*(BstarR-BstarL);
  E += (SB_neg * ST_pos * SL_neg * SR_pos) * tmpE;

  // SB>0
  tmpE = (SAR*EstarLLx-SAL*EstarRLx+SAR*SAL*(bRL-bLL))/(SAR-SAL);
  tmpE = SL_pos*ELL + SL_neg*SR_neg*ERL + SL_neg*SR_pos*tmpE;
  E += SB_pos * tmpE;

  // ST<0
  tmpE = (SAR*EstarLRx-SAL*EstarRRx+SAR*SAL*(bRR-bLR))/(SAR-SAL);
  tmpE = SL_pos*ELR + SL_neg*SR_neg*ERR + SL_neg*SR_pos*tmpE;
  E += (SB_neg * ST_neg) * tmpE;

  // SL>0
  tmpE = (SAT*EstarLLy-SAB*EstarLRy-SAT*SAB*(aLR-aLL))/(SAT-SAB);
  E += (SB_neg * ST_pos * SL_pos) * tmpE;

  // SR<0
  tmpE = (SAT*EstarRLy-SAB*EstarRRy-SAT*SAB*(aRR-aRL))/(SAT-SAB);
  E += (SB_neg * ST_pos * SL_neg * SR_neg) * tmpE;


  /*
  if(SB>ZERO_F) {
    if(SL>ZERO_F) {
      E=ELL;
    } else if(SR<ZERO_F) {
      E=ERL;
    } else {
      E=(SAR*EstarLLx-SAL*EstarRLx+SAR*SAL*(bRL-bLL))/(SAR-SAL);
    }
  } else if (ST<ZERO_F) {
    if(SL>ZERO_F) {
      E=ELR;
    } else if(SR<ZERO_F) {
      E=ERR;
    } else {
      E=(SAR*EstarLRx-SAL*EstarRRx+SAR*SAL*(bRR-bLR))/(SAR-SAL);
    }
  } else if (SL>ZERO_F) {
    E=(SAT*EstarLLy-SAB*EstarLRy-SAT*SAB*(aLR-aLL))/(SAT-SAB);
  } else if (SR<ZERO_F) {
    E=(SAT*EstarRLy-SAB*EstarRRy-SAT*SAB*(aRR-aRL))/(SAT-SAB);
  } else {
    E = (SAL*SAB*EstarRR-SAL*SAT*EstarRL - 
	 SAR*SAB*EstarLR+SAR*SAT*EstarLL)/(SAR-SAL)/(SAT-SAB) - 
      SAT*SAB/(SAT-SAB)*(AstarT-AstarB) + 
      SAR*SAL/(SAR-SAL)*(BstarR-BstarL);
  }
  */

  return E;

} // mag_riemann2d_hlld

/**
 * 2D magnetic riemann solver of type HLLD (mixed precision)
 *
 */
__DEVICE__ inline
real_t mag_riemann2d_hlld_mixed(real_t qLLRR[4][NVAR_MHD],
				real_t eLLRR[4])
{

  // alias reference to input arrays
  real_t (&qLL)[NVAR_MHD] = qLLRR[ILL];
  real_t (&qRL)[NVAR_MHD] = qLLRR[IRL];
  real_t (&qLR)[NVAR_MHD] = qLLRR[ILR];
  real_t (&qRR)[NVAR_MHD] = qLLRR[IRR];

  real_t &ELL = eLLRR[ILL];
  real_t &ERL = eLLRR[IRL];
  real_t &ELR = eLLRR[ILR];
  real_t &ERR = eLLRR[IRR];
  //real_t ELL,ERL,ELR,ERR;

  real_t &rLL=qLL[ID]; real_t &pLL=qLL[IP]; 
  real_t &uLL=qLL[IU]; real_t &vLL=qLL[IV]; 
  real_t &aLL=qLL[IA]; real_t &bLL=qLL[IB] ; real_t &cLL=qLL[IC];
  
  real_t &rLR=qLR[ID]; real_t &pLR=qLR[IP]; 
  real_t &uLR=qLR[IU]; real_t &vLR=qLR[IV]; 
  real_t &aLR=qLR[IA]; real_t &bLR=qLR[IB] ; real_t &cLR=qLR[IC];
  
  real_t &rRL=qRL[ID]; real_t &pRL=qRL[IP]; 
  real_t &uRL=qRL[IU]; real_t &vRL=qRL[IV]; 
  real_t &aRL=qRL[IA]; real_t &bRL=qRL[IB] ; real_t &cRL=qRL[IC];

  real_t &rRR=qRR[ID]; real_t &pRR=qRR[IP]; 
  real_t &uRR=qRR[IU]; real_t &vRR=qRR[IV]; 
  real_t &aRR=qRR[IA]; real_t &bRR=qRR[IB] ; real_t &cRR=qRR[IC];
  
  // Compute 4 fast magnetosonic velocity relative to x direction
  float cFastLLx = find_speed_fast<IX>(qLL);
  float cFastLRx = find_speed_fast<IX>(qLR);
  float cFastRLx = find_speed_fast<IX>(qRL);
  float cFastRRx = find_speed_fast<IX>(qRR);

  // Compute 4 fast magnetosonic velocity relative to y direction 
  float cFastLLy = find_speed_fast<IY>(qLL);
  float cFastLRy = find_speed_fast<IY>(qLR);
  float cFastRLy = find_speed_fast<IY>(qRL);
  float cFastRRy = find_speed_fast<IY>(qRR);
  
  // TODO : write a find_speed that computes the 2 speeds together (in
  // a single routine -> factorize computation of cFastLLx and cFastLLy

  float SL = FMIN4(uLL,uLR,uRL,uRR) - FMAX4(cFastLLx,cFastLRx,cFastRLx,cFastRRx);
  float SR = FMAX4(uLL,uLR,uRL,uRR) + FMAX4(cFastLLx,cFastLRx,cFastRLx,cFastRRx);
  float SB = FMIN4(vLL,vLR,vRL,vRR) - FMAX4(cFastLLy,cFastLRy,cFastRLy,cFastRRy);
  float ST = FMAX4(vLL,vLR,vRL,vRR) + FMAX4(cFastLLy,cFastLRy,cFastRLy,cFastRRy);

  /*ELL = uLL*bLL - vLL*aLL;
    ELR = uLR*bLR - vLR*aLR;
    ERL = uRL*bRL - vRL*aRL;
    ERR = uRR*bRR - vRR*aRR;*/
  
  float PtotLL = pLL + HALF_F * (aLL*aLL + bLL*bLL + cLL*cLL);
  float PtotLR = pLR + HALF_F * (aLR*aLR + bLR*bLR + cLR*cLR);
  float PtotRL = pRL + HALF_F * (aRL*aRL + bRL*bRL + cRL*cRL);
  float PtotRR = pRR + HALF_F * (aRR*aRR + bRR*bRR + cRR*cRR);
  
  float rcLLx = rLL * (uLL-SL); float rcRLx = rRL *(SR-uRL);
  float rcLRx = rLR * (uLR-SL); float rcRRx = rRR *(SR-uRR);
  float rcLLy = rLL * (vLL-SB); float rcLRy = rLR *(ST-vLR);
  float rcRLy = rRL * (vRL-SB); float rcRRy = rRR *(ST-vRR);

  float ustar = (rcLLx*uLL + rcLRx*uLR + rcRLx*uRL + rcRRx*uRR +
		  (PtotLL - PtotRL + PtotLR - PtotRR) ) / (rcLLx + rcLRx + 
							   rcRLx + rcRRx);
  float vstar = (rcLLy*vLL + rcLRy*vLR + rcRLy*vRL + rcRRy*vRR +
		  (PtotLL - PtotLR + PtotRL - PtotRR) ) / (rcLLy + rcLRy + 
							   rcRLy + rcRRy);
  
  float rstarLLx = rLL * (SL-uLL) / (SL-ustar);
  float BstarLL  = bLL * (SL-uLL) / (SL-ustar);
  float rstarLLy = rLL * (SB-vLL) / (SB-vstar); 
  float AstarLL  = aLL * (SB-vLL) / (SB-vstar);
  float rstarLL  = rLL * (SL-uLL) / (SL-ustar) 
    *                     (SB-vLL) / (SB-vstar);
  float EstarLLx = ustar * BstarLL - vLL   * aLL;
  float EstarLLy = uLL   * bLL     - vstar * AstarLL;
  float EstarLL  = ustar * BstarLL - vstar * AstarLL;
  
  float rstarLRx = rLR * (SL-uLR) / (SL-ustar); 
  float BstarLR  = bLR * (SL-uLR) / (SL-ustar);
  float rstarLRy = rLR * (ST-vLR) / (ST-vstar); 
  float AstarLR  = aLR * (ST-vLR) / (ST-vstar);
  float rstarLR  = rLR * (SL-uLR) / (SL-ustar) * (ST-vLR) / (ST-vstar);
  float EstarLRx = ustar * BstarLR - vLR   * aLR;
  float EstarLRy = uLR   * bLR     - vstar * AstarLR;
  float EstarLR  = ustar * BstarLR - vstar * AstarLR;

  float rstarRLx = rRL * (SR-uRL) / (SR-ustar); 
  float BstarRL  = bRL * (SR-uRL) / (SR-ustar);
  float rstarRLy = rRL * (SB-vRL) / (SB-vstar); 
  float AstarRL  = aRL * (SB-vRL) / (SB-vstar);
  float rstarRL  = rRL * (SR-uRL) / (SR-ustar) * (SB-vRL) / (SB-vstar);
  float EstarRLx = ustar * BstarRL - vRL   * aRL;
  float EstarRLy = uRL   * bRL     - vstar * AstarRL;
  float EstarRL  = ustar * BstarRL - vstar * AstarRL;
  
  float rstarRRx = rRR * (SR-uRR) / (SR-ustar); 
  float BstarRR  = bRR * (SR-uRR) / (SR-ustar);
  float rstarRRy = rRR * (ST-vRR) / (ST-vstar); 
  float AstarRR  = aRR * (ST-vRR) / (ST-vstar);
  float rstarRR  = rRR * (SR-uRR) / (SR-ustar) * (ST-vRR) / (ST-vstar);
  float EstarRRx = ustar * BstarRR - vRR   * aRR;
  float EstarRRy = uRR   * bRR     - vstar * AstarRR;
  float EstarRR  = ustar * BstarRR - vstar * AstarRR;

  float calfvenL = FMAX5(FABS(aLR)/SQRT(rstarLRx), FABS(AstarLR)/SQRT(rstarLR), 
			  FABS(aLL)/SQRT(rstarLLx), FABS(AstarLL)/SQRT(rstarLL), 
			  gParams.smallc);
  float calfvenR = FMAX5(FABS(aRR)/SQRT(rstarRRx), FABS(AstarRR)/SQRT(rstarRR),
			  FABS(aRL)/SQRT(rstarRLx), FABS(AstarRL)/SQRT(rstarRL), 
			  gParams.smallc);
  float calfvenB = FMAX5(FABS(bLL)/SQRT(rstarLLy), FABS(BstarLL)/SQRT(rstarLL), 
			  FABS(bRL)/SQRT(rstarRLy), FABS(BstarRL)/SQRT(rstarRL), 
			  gParams.smallc);
  float calfvenT = FMAX5(FABS(bLR)/SQRT(rstarLRy), FABS(BstarLR)/SQRT(rstarLR), 
			  FABS(bRR)/SQRT(rstarRRy), FABS(BstarRR)/SQRT(rstarRR), 
			  gParams.smallc);

  float SAL = FMIN(ustar - calfvenL, (float) ZERO_F); 
  float SAR = FMAX(ustar + calfvenR, (float) ZERO_F);
  float SAB = FMIN(vstar - calfvenB, (float) ZERO_F); 
  float SAT = FMAX(vstar + calfvenT, (float) ZERO_F);

  float AstarT = (SAR*AstarRR - SAL*AstarLR) / (SAR-SAL); 
  float AstarB = (SAR*AstarRL - SAL*AstarLL) / (SAR-SAL);
  
  float BstarR = (SAT*BstarRR - SAB*BstarRL) / (SAT-SAB); 
  float BstarL = (SAT*BstarLR - SAB*BstarLL) / (SAT-SAB);

  // finally get emf E
  float E=0, tmpE=0;

  // the following part is slightly different from the original fortran
  // code since it has to much different branches
  // which generate to much branch divergence in CUDA !!!

  // compute sort of boolean (don't know if signbit is available)
  int SB_pos = (int) (1+COPYSIGN(ONE_F,SB))/2, SB_neg = 1-SB_pos;
  int ST_pos = (int) (1+COPYSIGN(ONE_F,ST))/2, ST_neg = 1-ST_pos;
  int SL_pos = (int) (1+COPYSIGN(ONE_F,SL))/2, SL_neg = 1-SL_pos;
  int SR_pos = (int) (1+COPYSIGN(ONE_F,SR))/2, SR_neg = 1-SR_pos;

  // else
  tmpE = (SAL*SAB*EstarRR-SAL*SAT*EstarRL - 
	  SAR*SAB*EstarLR+SAR*SAT*EstarLL)/(SAR-SAL)/(SAT-SAB) - 
    SAT*SAB/(SAT-SAB)*(AstarT-AstarB) + 
    SAR*SAL/(SAR-SAL)*(BstarR-BstarL);
  E += (SB_neg * ST_pos * SL_neg * SR_pos) * tmpE;

  // SB>0
  tmpE = (SAR*EstarLLx-SAL*EstarRLx+SAR*SAL*(bRL-bLL))/(SAR-SAL);
  tmpE = SL_pos*ELL + SL_neg*SR_neg*ERL + SL_neg*SR_pos*tmpE;
  E += SB_pos * tmpE;

  // ST<0
  tmpE = (SAR*EstarLRx-SAL*EstarRRx+SAR*SAL*(bRR-bLR))/(SAR-SAL);
  tmpE = SL_pos*ELR + SL_neg*SR_neg*ERR + SL_neg*SR_pos*tmpE;
  E += (SB_neg * ST_neg) * tmpE;

  // SL>0
  tmpE = (SAT*EstarLLy-SAB*EstarLRy-SAT*SAB*(aLR-aLL))/(SAT-SAB);
  E += (SB_neg * ST_pos * SL_pos) * tmpE;

  // SR<0
  tmpE = (SAT*EstarRLy-SAB*EstarRRy-SAT*SAB*(aRR-aRL))/(SAT-SAB);
  E += (SB_neg * ST_pos * SL_neg * SR_neg) * tmpE;


  /*
  if(SB>ZERO_F) {
    if(SL>ZERO_F) {
      E=ELL;
    } else if(SR<ZERO_F) {
      E=ERL;
    } else {
      E=(SAR*EstarLLx-SAL*EstarRLx+SAR*SAL*(bRL-bLL))/(SAR-SAL);
    }
  } else if (ST<ZERO_F) {
    if(SL>ZERO_F) {
      E=ELR;
    } else if(SR<ZERO_F) {
      E=ERR;
    } else {
      E=(SAR*EstarLRx-SAL*EstarRRx+SAR*SAL*(bRR-bLR))/(SAR-SAL);
    }
  } else if (SL>ZERO_F) {
    E=(SAT*EstarLLy-SAB*EstarLRy-SAT*SAB*(aLR-aLL))/(SAT-SAB);
  } else if (SR<ZERO_F) {
    E=(SAT*EstarRLy-SAB*EstarRRy-SAT*SAB*(aRR-aRL))/(SAT-SAB);
  } else {
    E = (SAL*SAB*EstarRR-SAL*SAT*EstarRL - 
	 SAR*SAB*EstarLR+SAR*SAT*EstarLL)/(SAR-SAL)/(SAT-SAB) - 
      SAT*SAB/(SAT-SAB)*(AstarT-AstarB) + 
      SAR*SAL/(SAR-SAL)*(BstarR-BstarL);
  }
  */

  return E;

} // mag_riemann2d_hlld_mixed

/**
 * Compute emf from qEdge state vector via a 2D magnetic Riemann
 * solver (see routine cmp_mag_flux in DUMSES).
 *
 * @param[in] qEdge array containing input states qRT, qLT, qRB, qLB
 * @param[in] xPos x position in space (only needed for shearing box correction terms).
 * @return emf 
 *
 * template parameters:
 *
 * @tparam emfDir plays the role of xdim/lor in DUMSES routine
 * cmp_mag_flx, i.e. define which EMF will be computed (how to define
 * parallel/orthogonal velocity). emfDir identifies the orthogonal direction.
 *
 * \note the global parameter magRiemannSolver is used to choose the
 * 2D magnetic Riemann solver.
 *
 * TODO: make xPos parameter non-optional
 */
template <EmfDir emfDir>
__DEVICE__
real_t compute_emf(real_t qEdge [4][NVAR_MHD], real_t xPos=0)
{
  
  // define alias reference to input arrays
  real_t (&qRT)[NVAR_MHD] = qEdge[IRT];
  real_t (&qLT)[NVAR_MHD] = qEdge[ILT];
  real_t (&qRB)[NVAR_MHD] = qEdge[IRB];
  real_t (&qLB)[NVAR_MHD] = qEdge[ILB];

  // defines alias reference to intermediate state before applying a
  // magnetic Riemann solver
  real_t qLLRR[4][NVAR_MHD];
  real_t (&qLL)[NVAR_MHD] = qLLRR[ILL];
  real_t (&qRL)[NVAR_MHD] = qLLRR[IRL];
  real_t (&qLR)[NVAR_MHD] = qLLRR[ILR];
  real_t (&qRR)[NVAR_MHD] = qLLRR[IRR];
  
  // density
  qLL[ID] = qRT[ID];
  qRL[ID] = qLT[ID];
  qLR[ID] = qRB[ID];
  qRR[ID] = qLB[ID];

  // pressure
  // ISOTHERMAL
  real_t cIso = ::gParams.cIso;
  if (cIso > 0) {
    qLL[IP] = qLL[ID]*cIso*cIso;
    qRL[IP] = qRL[ID]*cIso*cIso;
    qLR[IP] = qLR[ID]*cIso*cIso;
    qRR[IP] = qRR[ID]*cIso*cIso;
  } else {
    qLL[IP] = qRT[IP];
    qRL[IP] = qLT[IP];
    qLR[IP] = qRB[IP];
    qRR[IP] = qLB[IP];
  }

  // iu, iv : parallel velocity indexes
  // iw     : orthogonal velocity index
  // ia, ib, ic : idem for magnetic field
  int iu, iv, iw, ia, ib, ic;
  if (emfDir == EMFZ) {
    iu = IU; iv = IV; iw = IW;
    ia = IA; ib = IB, ic = IC;
  } else if (emfDir == EMFY) {
    iu = IW; iv = IU; iw = IV;
    ia = IC; ib = IA, ic = IB;
  } else { // emfDir == EMFX
    iu = IV; iv = IW; iw = IU;
    ia = IB; ib = IC, ic = IA;
  }

  // First parallel velocity 
  qLL[IU] = qRT[iu];
  qRL[IU] = qLT[iu];
  qLR[IU] = qRB[iu];
  qRR[IU] = qLB[iu];
  
  // Second parallel velocity 
  qLL[IV] = qRT[iv];
  qRL[IV] = qLT[iv];
  qLR[IV] = qRB[iv];
  qRR[IV] = qLB[iv];
    
  // First parallel magnetic field (enforce continuity)
  qLL[IA] = HALF_F * ( qRT[ia] + qLT[ia] );
  qRL[IA] = HALF_F * ( qRT[ia] + qLT[ia] );
  qLR[IA] = HALF_F * ( qRB[ia] + qLB[ia] );
  qRR[IA] = HALF_F * ( qRB[ia] + qLB[ia] );
  
  // Second parallel magnetic field (enforce continuity)
  qLL[IB] = HALF_F * ( qRT[ib] + qRB[ib] );
  qRL[IB] = HALF_F * ( qLT[ib] + qLB[ib] );
  qLR[IB] = HALF_F * ( qRT[ib] + qRB[ib] );
  qRR[IB] = HALF_F * ( qLT[ib] + qLB[ib] );
  
  // Orthogonal velocity 
  qLL[IW] = qRT[iw];
  qRL[IW] = qLT[iw];
  qLR[IW] = qRB[iw];
  qRR[IW] = qLB[iw];
  
  // Orthogonal magnetic Field
  qLL[IC] = qRT[ic];
  qRL[IC] = qLT[ic];
  qLR[IC] = qRB[ic];
  qRR[IC] = qLB[ic];
  
  // Compute final fluxes
  
  // vx*by - vy*bx at the four edge centers
  real_t eLLRR[4];
  real_t &ELL = eLLRR[ILL];
  real_t &ERL = eLLRR[IRL];
  real_t &ELR = eLLRR[ILR];
  real_t &ERR = eLLRR[IRR];

  ELL = qLL[IU]*qLL[IB] - qLL[IV]*qLL[IA];
  ERL = qRL[IU]*qRL[IB] - qRL[IV]*qRL[IA];
  ELR = qLR[IU]*qLR[IB] - qLR[IV]*qLR[IA];
  ERR = qRR[IU]*qRR[IB] - qRR[IV]*qRR[IA];

  real_t emf=0;
  // mag_riemann2d<>
  if (gParams.magRiemannSolver == MAG_HLLD) {
    emf = mag_riemann2d_hlld(qLLRR, eLLRR);
  } else if (gParams.magRiemannSolver == MAG_HLLA) {
    emf = mag_riemann2d_hlla(qLLRR, eLLRR);
  } else if (gParams.magRiemannSolver == MAG_HLLF) {
    emf = mag_riemann2d_hllf(qLLRR, eLLRR);
  } else if (gParams.magRiemannSolver == MAG_LLF) {
    emf = mag_riemann2d_llf(qLLRR, eLLRR);
  }

  /* upwind solver in case of the shearing box */
  if ( /* cartesian */ (::gParams.Omega0>0) /* and not fargo */ ) {
    if (emfDir==EMFX) {
      real_t shear = -1.5 * ::gParams.Omega0 * xPos;
      if (shear>0) {
	emf += shear * qLL[IB];
      } else {
	emf += shear * qRR[IB];
      }
    }
    if (emfDir==EMFZ) {
      real_t shear = -1.5 * ::gParams.Omega0 * (xPos - ::gParams.dx/2);
      if (shear>0) {
	emf -= shear * qLL[IA];
      } else {
	emf -= shear * qRR[IA];
      }
    }
  }

  return emf;

} // compute_emf

#endif // RIEMANN_MHD_H_
