/**
 * \file positiveScheme.h
 * \brief Intermediate routines used in the positive schemetime step evolution.
 *
 * Most of the routine defined here are translated/adapted from the
 * original code found in article by Lax and Liu, "Solution of the two-dimensional
 * Riemann problems of gas dynamics by positive schemes", SIMA
 * J. Sci. Comput., vol 19, pp 319-340, 1998.
 *
 * \author Pierre Kestener
 *
 * $Id: positiveScheme.h 1784 2012-02-21 10:34:58Z pkestene $
 */
#ifndef POSITIVE_SCHEME_H_
#define POSITIVE_SCHEME_H_

#include "constants.h"
#include "stdlib.h"

template <class T> const T max ( const T a, const T b ) {
  return (b<a)?a:b;
}

////////////////////////////////////////////////////////////////////////////
/** 
 * \fn void limiter(float dw, float dwup, float &phi0, float &phi1, int k)
 * \brief Evaluate limiters
 * 
 * Evaluate 2 limiters \f$ \phi_0 \f$ and \f$ \phi_1 \f$ from
 * \f$ \theta = dw/dwup \f$
 * 
 * @param dw 
 * @param dwup 
 * @param phi0 
 * @param phi1 
 */
inline __DEVICE__
void limiter(float dw, float dwup, float &phi0, float &phi1, int k)
{
  float theta, phi;
  
  //--------Superbee---------------------------------------------------------
  phi0=0.0f;
  if(dw == 0.0f && dwup>0.0f)
    phi0=2.0f;
  if(dw*dwup>0.0f) {
    theta=dwup/dw;
    
    if(theta<=0.5f) {
      phi0=2.0f*theta;
    } else if(theta<=1.0f && theta>0.5f) {
      phi0=1.0f;
    } else if(theta<=2.0f && theta>1.0f) {
      phi0=theta;
    } else {
      phi0=2.0f;
    }
  }
  
  //-------VanLeer---------------------------------------------------------
  phi=0.0f;
  if(dw==0 && dwup>0)
    phi=2.0f;
  if(dw*dwup>0) {
    theta=dwup/dw;
    phi=2.0f*theta/(1.0f+theta);
  }
  
  //-------MinMod----------------------------------------------------------
  phi1=0.0f;
  if(dw==0.0f && dwup>0.0f)
    phi1=1.0f;
  if(dw*dwup>0) {
    phi1=1.0f;
    if(dwup/dw<=1.0f) {
      phi1=dwup/dw;
    }
  }
  
  //-----------------------------------------------------------------------
  // k=0 or k=3 : phi0 -> VanLeer
  // k=1 or k=2 : phi0 -> Superbee
  if(k==0 || k==3) {
    phi0=phi;
  }
  
} // limiter 

////////////////////////////////////////////////////////////////////////////
/** 
 * \fn void eigs(float up[NVAR], float um[NVAR], float
 * (&r)[NVAR][NVAR], float (&ri)[NVAR][NVAR], float (&eig)[NVAR])
 * \brief Compute right, left eigenvectors of Roe Matrix and its eigenvalues
 * 
 * @param up  : input state \f$ U_{j+1}\f$
 * @param um  : input state \f$ U_{j}\f$
 * @param r   : output right eigenvectors
 * @param ri  : output left eigenvectors
 * @param eig : output eigenvalues
 *
 * NOTE : only NVAR=NVAR_2D is currently supported
 */
template <NvarSimulation NVAR>
inline __DEVICE__
int eigs(float up[NVAR], float um[NVAR], float (&r)[NVAR][NVAR], float (&ri)[NVAR][NVAR], float (&eig)[NVAR])
{
  float r1  = fmaxf(um[0], ::gParams.smallr);
  float u1  = um[1]/r1;
  float v1  = um[2]/r1;
  float ek1 = 0.5f*(u1*u1+v1*v1);
  float H1  = (um[3]+(::gParams.gamma0-1.0f)*(um[3]-ek1*r1))/r1;
  
  float r2  = fmaxf(up[0], ::gParams.smallr);
  float u2  = up[1]/r2;
  float v2  = up[2]/r2;
  float ek2 = 0.5f*(u2*u2+v2*v2);
  float H2  = (up[3]+(::gParams.gamma0-1.0f)*(up[3]-ek2*r2))/r2;

  float w1  = (sqrt(r1)      +sqrt(r2));
  float u   = (sqrt(r1)*u1   +sqrt(r2)*u2)/w1;
  float v   = (sqrt(r1)*v1   +sqrt(r2)*v2)/w1;
  float H   = (sqrt(r1)*H1   +sqrt(r2)*H2)/w1;

  //float p,rho;
  //rho = (r1+r2)*0.5f;
  float q2 = u*u+v*v;
  float c = (::gParams.gamma0-1.0f)*(H-0.5f*q2);
  if (c<0) {
    c = ::gParams.smallc;
  } else {
    c = sqrt(c);
  }

  //float c  = sqrt(::gParams.gamma0*(::gParams.gamma0-1.0f)*(Er-0.5f*q2));
  //eos(rho, ei-0.5*q2,p,c);

#ifndef __CUDACC__
  if (isnan(c)) {
    printf("sound speed is not a number !!\n");
    return EXIT_FAILURE;
  }
  if (isnan(u)) {
    printf("velocity u  is not a number !!\n");
    return EXIT_FAILURE;
  }
#endif // __CUDACC__
  
  r[0][0] = 1.0f;
  r[1][0] = u-c;
  r[2][0] = v;
  r[3][0] = H-u*c;
  r[0][1] = 0.0f;
  r[1][1] = 0.0f;
  r[2][1] = 1.0f;
  r[3][1] = v;
  r[0][2] = 1.0f;
  r[1][2] = u;
  r[2][2] = v;
  r[3][2] = 0.5f*q2;
  r[0][3] = 1.0f;
  r[1][3] = u+c;
  r[2][3] = v;
  r[3][3] = H+u*c;

  float b1 = 1.0f/(H-0.5f*q2);
  //float b1 = (::gParams.gamma0-1)/(::gParams.gamma0)*rho/p; 
  b1 = fmaxf(b1, 1e-10);
  float b2 = 0.5f*q2*b1;
#ifndef __CUDACC__
  if (isnan(b1) || isinf(b1)) {
    printf("%.12f %.12f %f %f %f %f %f %f %f %f %f\n",b1,r1,r2,H,q2,u,v, u1, u2, v1, v2);
    return EXIT_FAILURE;
  }
  if (isnan(b2) || isinf(b2)) {
    printf("%.12f %.12f %f %f %f %f %f %f %f\n",b2,q2,u,v,b1,r1,r2,H,q2);
    return EXIT_FAILURE;
  }
#endif // __CUDACC__

  ri[0][0] =  0.5f*(b2+u/c);
  ri[0][1] = -0.5f/c-0.5f*b1*u;
  ri[0][2] = -0.5f*b1*v;
  ri[0][3] =  0.5f*b1;
  ri[1][0] = -v;
  ri[1][1] = 0.0f;
  ri[1][2] = 1.0f;
  ri[1][3] = 0.0f;
  ri[2][0] = 1.0f-b2;
  ri[2][1] = b1*u;
  ri[2][2] = b1*v;
  ri[2][3] = -b1;
  ri[3][0] = 0.5f*(b2-u/c);
  ri[3][1] = 0.5f/c-0.5f*b1*u;
  ri[3][2] =-0.5f*b1*v;
  ri[3][3] = 0.5f*b1;

  eig[0]=u-c;
  eig[1]=u;
  eig[2]=u;
  eig[3]=u+c;
  
  return EXIT_SUCCESS;

} // eigs

////////////////////////////////////////////////////////////////////////////
/** 
 * \fn void central_diff_flux(float up[NVAR], float um[NVAR], float (&fc)[NVAR])
 * \brief Compute central differencing flux
 * 
 * Compute \f$ fc = 0.5*(F(U_{j+1})+F(U_{j})) \f$
 *
 * @param up : input flux \f$ U_{j+1} \f$
 * @param um : input flux \f$ U_{j} \f$
 * @param fc : central flux output
 *
 * NOTE : only NVAR=NVAR_2D is currently supported
 */
template <NvarSimulation NVAR>
inline __DEVICE__
void central_diff_flux(float up[NVAR], float um[NVAR], float (&fc)[NVAR])
{
  float rleft  = fmaxf(um[ID], ::gParams.smallr);
  float rright = fmaxf(up[ID], ::gParams.smallr);
  float pleft  = (::gParams.gamma0-1.0f)*(um[IP]-0.5f*(um[IU]*um[IU]+um[IV]*um[IV])/rleft);
  float pright = (::gParams.gamma0-1.0f)*(up[IP]-0.5f*(up[IU]*up[IU]+up[IV]*up[IV])/rright);

  fc[0] = 0.5f * ( um[1]+up[1] );
  fc[1] = 0.5f * ( um[1]*um[1]/rleft+pleft +  up[1]*up[1]/rright+pright );
  fc[2] = 0.5f * ( um[1]*um[2]/rleft       +  up[1]*up[2]/rright        );
  fc[3] = 0.5f * ((um[3]+pleft)*um[1]/rleft+ (up[3]+pright)*up[1]/rright);
} // central_diff_flux

////////////////////////////////////////////////////////////////////////////
/** 
 * \fn void diffusive_flux(float up[NVAR], float um[NVAR], float
 * du[NVAR], float dup[NVAR], float dum[NVAR], float df[NVAR])
 * \brief Compute Diffusive flux
 * 
 * \note See article Lax,Liu "Solution to two-dimensional Riemann
 * problems of gas dynamics by positive schemes, SIAM J. Sci. Comput.,
 * Vol 19, p. 319
 *
 * @param up 
 * @param um 
 * @param du 
 * @param dup 
 * @param dum 
 * @param df : output
 *
 * NOTE : only NVAR=NVAR_2D is currently supported
 */
template <NvarSimulation NVAR>
inline __DEVICE__
int diffusive_flux(float up[NVAR], float um[NVAR], float du[NVAR], float dup[NVAR],
		    float dum[NVAR], float (&df)[NVAR])
{
  // local variables
  float dw[NVAR];
  float dwf[NVAR];
  float r[NVAR][NVAR];
  float ri[NVAR][NVAR];
  float eig[NVAR];

  if (eigs<NVAR>(up,um,r,ri,eig))
    return 1;
  float mu=fmaxf(fabs(eig[0]),fabs(eig[3]));

#ifndef __CUDACC__
  if (isnan(eig[1]))
    printf("eig[1] is not a number !!!\n");
  if (isnan(eig[0]))
    printf("eig[0] is not a number !!!\n");
#endif // __CUDACC__

  float dwup;
  for (int k=0; k<NVAR; ++k) {

    float phi0, phi1;
    //float dwup;

    dw[k] = ri[k][0]*du[0] +ri[k][1]*du[1] +ri[k][2]*du[2] +ri[k][3]*du[3];
    dwup  = ri[k][0]*dup[0]+ri[k][1]*dup[1]+ri[k][2]*dup[2]+ri[k][3]*dup[3];
    if (eig[k]>=0.0f) {
      dwup=ri[k][0]*dum[0]+ri[k][1]*dum[1]+ri[k][2]*dum[2]+ri[k][3]*dum[3];
    }

    // compute limiter coefficient phi0, phi1
    limiter(dw[k],dwup,phi0,phi1,k);

    dwf[k]=-0.5f*(gParams.ALPHA*(1.0f-phi0)*fabs(eig[k])
		  +gParams.BETA*(1.0f-phi1)*mu          )*dw[k];
  }
  
  // compute diffusive flux
  for (int k=0; k<NVAR; ++k) {
    df[k]=
      r[k][0]*dwf[0]+
      r[k][1]*dwf[1]+
      r[k][2]*dwf[2]+
      r[k][3]*dwf[3];
  }
  
#ifndef __CUDACC__
  if (isnan(df[0])) {
    printf("Arghhh....\n");
    printf("%f %f %.12f %.12f %f %f %f %f\n",ri[0][0],ri[0][1],ri[0][2],ri[0][3],dwf[0],dwf[1],dwf[2],dwf[3]);
    printf("du   %f %f %f %f\n",du[0],du[1],du[2],du[3]);
    printf("dw   %f %f %f %f\n",dw[0],dw[1],dw[2],dw[3]);
    printf("dwup %f\n",dwup);
    printf("dwf  %f %f %f %f\n",dwf[0],dwf[1],dwf[2],dwf[3]);
  }
#endif // __CUDACC__

  return 0;

} // diffusive_flux



#endif // POSITIVE_SCHEME_H_
