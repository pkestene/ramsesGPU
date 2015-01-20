/**
 * \file constoprim.h
 * \brief Commonly used function to compute primitive variables from
 * conservatibes variables.
 *
 * \author F. Chateau, P. Kestener 
 *
 * $Id: constoprim.h 3587 2014-11-01 21:57:10Z pkestene $
 */
#ifndef CONSTOPRIM_H_
#define CONSTOPRIM_H_

#include "real_type.h"
#include "constants.h"

/**
 * compute pressure p and speed of sound c, from density rho and
 * internal energy eint using the "calorically perfect gas" equation
 * of state : \f$ eint=\frac{p}{\rho (\gamma-1)} \f$
 * Recall that \f$ \gamma \f$ is equal to the ratio of specific heats \f$ \left[
 * c_p/c_v \right] \f$.
 * 
 * @param[in]  rho  density
 * @param[in]  eint internal energy per mass unit
 * @param[out] p    pressure
 * @param[out] c    speed of sound
 */
inline __DEVICE__ 
void eos(real_t rho, real_t eint, real_t& p, real_t& c)
{
  p = FMAX((::gParams.gamma0 - 1.0f) * rho * eint, rho * ::gParams.smallp);
  c = SQRT(::gParams.gamma0 * p / rho);
}

/**
 * conservative (rho, rho*u, rho*v, e) to primitive variables (rho,
 * u,v,p)
 * @param[in]  u  conservative variables
 * @param[out] q  primitive    variables
 * @param[out] c  local speed of sound
 */
inline __DEVICE__ 
void constoprim_2D(real_t u[NVAR_2D], real_t (&q)[NVAR_2D], real_t& c)
{
  q[ID] = FMAX(u[ID], ::gParams.smallr);

  q[IU] = u[IU] / q[ID];
  q[IV] = u[IV] / q[ID];

  // kinetic energy per mass unit
  real_t eken = 0.5f * (q[IU] * q[IU] + q[IV] * q[IV]);

  // compute pressure
  if (::gParams.cIso > 0) { // isothermal eos : P = c_s^2 * \rho
    
    q[IP] = q[ID] * (::gParams.cIso) * (::gParams.cIso);
    c     =  ::gParams.cIso;

  } else {
    
    // internal energy = total energy - kinetic energy (per mass unit)
    real_t eint = u[IP] / q[ID] - eken;
    if (eint < 0) {
      PRINTF("hydro eint < 0  : e %f eken %f d %f u %f v %f\n",u[IP],eken,u[ID],u[IU],u[IV]);
    }
    // use perfect gas equation of state to compute P
    eos(q[ID], eint, q[IP], c);

  } // end compute pressure

} // constoprim_2D

/**
 * conservative (rho, rho*u, rho*v, rho*w, e) to primitive variables (rho,
 * u,v,w, p).
 *
 * @param[in]  u  conservative variables
 * @param[out] q  primitive    variables
 * @param[out] c  local speed of sound
 */
inline __DEVICE__ 
void constoprim_3D(real_t u[NVAR_3D], real_t (&q)[NVAR_3D], real_t& c)
{

  q[ID] = FMAX(u[ID], ::gParams.smallr);

  q[IU] = u[IU] / q[ID];
  q[IV] = u[IV] / q[ID];
  q[IW] = u[IW] / q[ID];
  
  // kinetic energy
  real_t eken = 0.5f * (q[IU] * q[IU] + q[IV] * q[IV] + q[IW] * q[IW]);

  // compute pressure
  if (::gParams.cIso > 0) { // isothermal eos : P = c_s^2 * \rho
    
    q[IP] = q[ID] * (::gParams.cIso) * (::gParams.cIso);
    c     =  ::gParams.cIso;
    
  } else {

    real_t eint = u[IP] / q[ID] - eken;
    if (eint < 0) {
      PRINTF("hydro eint < 0  : e %f eken %f d %f u %f v %f w %f\n",u[IP],eken,u[ID],u[IU],u[IV],u[IW]);
    }
    // use perfect gas equation of state to compute P
    eos(q[ID], eint, q[IP], c);

  } // end compute pressure

} // constoprim_3D

/**
 * MHD convert conservative to primitive variables.
 * The main difference with hydro is that we need neighbors data because 
 * we need to compute cell-centered magnetic field from face-centered values.
 *
 * \sa DUMSES/src/ctoprim.f90
 * The main difference with DUMSES is that we do not store right face-centered 
 * variables, so we absolutely need a ghostWidth of 3.
 *
 * Please also remember that in present code, all functions are pointwise 
 * (loop over the entire domain are handled in the calling routine or in the 
 * CUDA kernel).
 *
 * magFieldNeighbors is a 3 component array, containing left face-centered 
 * magnetic field BX in cell located at i+1 (resp. BY in cell at j+1 and BZ
 * in cell at k+1). 
 *
 * @param[in]  u                 conservative variables
 * @param[in]  magFieldNeighbors face-centered magnetic fields in neighboring cells.
 * @param[out] q                 primitive variables
 * @param[out] c                 local speed of sound
 * @param[in]  dt                time step (needed for predictor computations)
 */
inline __DEVICE__ 
bool constoprim_mhd(real_t u[NVAR_MHD], 
		    real_t magFieldNeighbors[3], 
		    real_t (&q)[NVAR_MHD], 
		    real_t& c, 
		    real_t dt)
{
  bool status = true;
  
  // compute density
  q[ID] = FMAX(u[ID], ::gParams.smallr);

  // compute velocities
  q[IU] = u[IU] / q[ID];
  q[IV] = u[IV] / q[ID];
  q[IW] = u[IW] / q[ID];

  // compute CELL-CENTERED magnetic field
  q[IA] = HALF_F * ( u[IA] + magFieldNeighbors[0] );
  q[IB] = HALF_F * ( u[IB] + magFieldNeighbors[1] );
  q[IC] = HALF_F * ( u[IC] + magFieldNeighbors[2] );

  // compute specific kinetic energy and magnetic energy
  real_t eken = HALF_F * (q[IU] * q[IU] + q[IV] * q[IV] + q[IW] * q[IW]);
  real_t emag = HALF_F * (q[IA] * q[IA] + q[IB] * q[IB] + q[IC] * q[IC]);

  // compute pressure

  if (::gParams.cIso > 0) { // isothermal

    q[IP] = q[ID] * (::gParams.cIso) * (::gParams.cIso);
    c     =  ::gParams.cIso;

  } else {

    real_t eint = (u[IP] - emag) / q[ID] - eken;
    if (eint < 0) {
      //printf("MHD eint < 0  : e %f eken %f emag %f d %f u %f v %f w %f\n",u[IP],eken,emag,u[ID],u[IU],u[IV],u[IW]);
      status = false;
    }

    q[IP] = FMAX((::gParams.gamma0-ONE_F) * q[ID] * eint, q[ID] * ::gParams.smallp);
  
    if (q[IP] < 0) {
      PRINTF("MHD pressure neg !!!\n");
    }

    // compute speed of sound (should be removed as it is useless, hydro
    // legacy)
    c = SQRT(::gParams.gamma0 * q[IP] / q[ID]);
  }


  // Coriolis force predictor step
  if (/* cartesian */ ::gParams.Omega0 > 0) {
    real_t dvx= 2.0 * ::gParams.Omega0 * q[IV];
    real_t dvy=-0.5 * ::gParams.Omega0 * q[IU];
    q[IU] += dvx*dt*HALF_F;
    q[IV] += dvy*dt*HALF_F;
  }

  return status;

} // constoprim_mhd

// inline __DEVICE__ 
// void constoprim_3D(real_t u[NVAR_3D], real_t (&q)[NVAR_3D], real_t& c)
// {
//   q[ID] = FMAX(u[ID], ::gParams.smallr);
//   //q[ID] = FMAX(u[ID], 1e-10);
//   q[IU] = u[IU] / q[ID];
//   q[IV] = u[IV] / q[ID];
//   q[IW] = u[IW] / q[ID];
//   real_t eken = 0.5f * (q[IU] * q[IU] + q[IV] * q[IV] + q[IW] * q[IW]);
//   real_t e = u[IP] / q[ID] - eken;
//   eos(q[ID], e, q[IP], c);
// }

/**
 * primitive variables (rho, u, v, p) to conservative variables (rho,
 * rhou, rhov, E_total)
 * @param U       input (primitive variables) / output (conservative variables)
 * @param _gamma0 specific heat ratio
 */
inline
void primToCons_2D(real_t (&U)[NVAR_2D], real_t _gamma0)
{
      
  real_t rho = U[ID];
  real_t p   = U[IP];
  real_t u   = U[IU];
  real_t v   = U[IV];

  U[IU] *= rho; // rho*u
  U[IV] *= rho; // rho*v

  U[IP] = p/(_gamma0-1.0f) + rho*(u*u+v*v)*0.5f;

} // primToCons_2D

/**
 * primitive variables (rho, u, v, w, p) to conservative variables (rho,
 * rhou, rhov, rhow, E_total)
 * @param U       input (primitive variables) / output (conservative variables)
 * @param _gamma0 specific heat ratio
 */
inline
void primToCons_3D(real_t (&U)[NVAR_3D], real_t _gamma0)
{
      
  real_t rho = U[ID];
  real_t p   = U[IP];
  real_t u   = U[IU];
  real_t v   = U[IV];
  real_t w   = U[IW];

  U[IU] *= rho; // rho*u
  U[IV] *= rho; // rho*v
  U[IW] *= rho; // rho*w
  U[IP] = p/(_gamma0-1.0f) + rho*(u*u+v*v+w*w)*0.5f;
  
} // primToCons_3D

/** 
 * Compute primitive variables.
 * The template parameter is used to swap velocities.
 * This is only usefull for the CPU version, when this
 * function is used inside the Godunov procedure.
 *
 * Note : The swap specialization version is defined in implementation file.
 */
/*
template<int swap_v, unsigned int NVAR> 
inline __DEVICE__ 
void computePrimitives(real_t* U, int arraySize, int elemOffset, real_t& c, real_t (&q)[NVAR])
{
  real_t u[NVAR];
  int offset = elemOffset;

  int iu=IU;
  int iv=IV;
  if (swap_v) {
    iu = IV;
    iv = IU;
  }

  u[ID] = U[offset];  offset += arraySize;
  u[IP] = U[offset];  offset += arraySize;
  u[iu] = U[offset];  offset += arraySize;
  u[iv] = U[offset];  
  constoprim<NVAR>(u, q, c);
}
*/

/**
 * Compute hydro primitive variables in 2D
 */
inline __DEVICE__
void computePrimitives_0(real_t* U, int arraySize, int elemOffset, real_t& c, real_t (&q)[NVAR_2D])
{
  real_t u[NVAR_2D];
  int offset = elemOffset;
  u[ID] = U[offset];  offset += arraySize;
  u[IP] = U[offset];  offset += arraySize;
  u[IU] = U[offset];  offset += arraySize;
  u[IV] = U[offset];
  constoprim_2D(u, q, c);
}

/**
 * Compute hydro primitive variables in 2D (with swapping IU and IV)
 */
inline __DEVICE__
void computePrimitives_1(real_t* U, int arraySize, int elemOffset, real_t& c, real_t (&q)[NVAR_2D])
{
  real_t u[NVAR_2D];
  int offset = elemOffset;
  u[ID] = U[offset];  offset += arraySize;
  u[IP] = U[offset];  offset += arraySize;
  u[IV] = U[offset];  offset += arraySize;
  u[IU] = U[offset];
  constoprim_2D(u, q, c);
}

/**
 * Compute hydro primitive variables in 3D
 */
inline __DEVICE__
void computePrimitives_3D_0(real_t* U, int arraySize, int elemOffset, real_t& c, real_t (&q)[NVAR_3D])
{
  real_t u[NVAR_3D];
  int offset = elemOffset;
  u[ID] = U[offset];  offset += arraySize;
  u[IP] = U[offset];  offset += arraySize;
  u[IU] = U[offset];  offset += arraySize;
  u[IV] = U[offset];  offset += arraySize;
  u[IW] = U[offset];
  constoprim_3D(u, q, c);
}

/**
 * Compute hydro primitive variables in 3D after swapping IU and IV indexes
 */
inline __DEVICE__
void computePrimitives_3D_1(real_t* U, int arraySize, int elemOffset, real_t& c, real_t (&q)[NVAR_3D])
{
  real_t u[NVAR_3D];
  int offset = elemOffset;
  u[ID] = U[offset];  offset += arraySize;
  u[IP] = U[offset];  offset += arraySize;
  u[IV] = U[offset];  offset += arraySize;
  u[IU] = U[offset];  offset += arraySize;
  u[IW] = U[offset];
  constoprim_3D(u, q, c);
}

/**
 * Compute hydro primitive variables in 3D after swapping IU and IW indexes
 */
inline __DEVICE__
void computePrimitives_3D_2(real_t* U, int arraySize, int elemOffset, real_t& c, real_t (&q)[NVAR_3D])
{
  real_t u[NVAR_3D];
  int offset = elemOffset;
  u[ID] = U[offset];  offset += arraySize;
  u[IP] = U[offset];  offset += arraySize;
  u[IW] = U[offset];  offset += arraySize;
  u[IV] = U[offset];  offset += arraySize;
  u[IU] = U[offset];
  constoprim_3D(u, q, c);
}


////////////////////////////////////////////////////////////////
// MHD
////////////////////////////////////////////////////////////////

/**
 * Compute MHD primitive variables in 2D (cell-wise).
 *
 * \warning The main difference with hydro is that we need to access
 * magnetic field data in the current cell location, but also in
 * neighbors on the right !!
 *
 * \param[in] U pointer to MHD data array (real_t)
 * \param[in] dim array with 2 component, data array pitch and data array dimy
 * \param[in] elemOffset position in data array of current cell.
 * \param[out] c speed of sound
 * \param[out] q primitive variable vector
 * \param[in] dt time step (usefull to compute some predictor step).
 *
 */
inline __DEVICE__
void computePrimitives_MHD_2D(real_t* U, int dim[2], int elemOffset, real_t& c, real_t (&q)[NVAR_MHD], real_t dt)
{
  real_t u[NVAR_MHD];
  real_t magFieldNeighbors[3];
  int arraySize = dim[0]*dim[1];

  int offset = elemOffset;
  u[ID] = U[offset];  offset += arraySize;
  u[IP] = U[offset];  offset += arraySize;
  u[IU] = U[offset];  offset += arraySize;
  u[IV] = U[offset];  offset += arraySize;
  u[IW] = U[offset];  offset += arraySize;
  u[IA] = U[offset];  offset += arraySize;
  u[IB] = U[offset];  offset += arraySize;
  u[IC] = U[offset];

  // go to magnetic field components and get values from neighbors on the right
  offset = elemOffset + 5 * arraySize;
  magFieldNeighbors[IX] = U[offset+1];        offset += arraySize;
  magFieldNeighbors[IY] = U[offset+dim[0]];
  magFieldNeighbors[IZ] = ZERO_F;

  // return the primitive variable vector for current cell
  bool status = constoprim_mhd(u, magFieldNeighbors, q, c, dt);

#ifndef __CUDACC__
  if (!status) {
    //std::cout << "constoprim_mhd error @ x=" << elemOffset-(elemOffset/dim[0])*dim[0] << " y=" << elemOffset/dim[0] << std::endl;
  }
#endif

} // computePrimitives_MHD_2D

/**
 * Compute MHD primitive variables in 3D (cell-wise).
 *
 * \warning The main difference with hydro is that we need to access
 * magnetic field data in the current cell location, but also in
 * neighbors on the right !!
 *
 * \param[in] U pointer to MHD data array (real_t)
 * \param[in] dim array with 3 component, data array pitch, data array dimy and dimz
 * \param[in] elemOffset position in data array of current cell.
 * \param[out] c speed of sound
 * \param[out] q primitive variable vector
 * \param[in] dt time step (usefull to compute some predictor step).
 *
 */
inline __DEVICE__
void computePrimitives_MHD_3D(real_t* U, int dim[3], int elemOffset, real_t& c, real_t (&q)[NVAR_MHD], real_t dt)
{
  real_t u[NVAR_MHD];
  real_t magFieldNeighbors[3];
  int arraySize = dim[0]*dim[1]*dim[2];

  int offset = elemOffset;
  u[ID] = U[offset];  offset += arraySize;
  u[IP] = U[offset];  offset += arraySize;
  u[IU] = U[offset];  offset += arraySize;
  u[IV] = U[offset];  offset += arraySize;
  u[IW] = U[offset];  offset += arraySize;
  u[IA] = U[offset];  offset += arraySize;
  u[IB] = U[offset];  offset += arraySize;
  u[IC] = U[offset];

  // go to magnetic field components and get values from neighbors on the right
  offset = elemOffset + 5 * arraySize;
  magFieldNeighbors[IX] = U[offset+1];        offset += arraySize;
  magFieldNeighbors[IY] = U[offset+dim[0]];   offset += arraySize;
  magFieldNeighbors[IZ] = U[offset+dim[0]*dim[1]];

  // return the primitive variable vector for current cell
  constoprim_mhd(u, magFieldNeighbors, q, c, dt);

} // computePrimitives_MHD_3D

/**
 * get primitive variable vector state at some location.
 *
 * \param[in]  Q          Primitive variable raw array
 * \param[in]  arraySize  Domain size 
 * \param[in]  elemOffset offset to current location
 * \param[out] q          returned state vector of primitive variables
 * at current location.
 */
inline __DEVICE__
void getPrimitiveVector(real_t* Q, int arraySize, int elemOffset, real_t (&q)[NVAR_MHD]) {

  int offset = elemOffset;
  q[ID] = Q[offset];  offset += arraySize;
  q[IP] = Q[offset];  offset += arraySize;
  q[IU] = Q[offset];  offset += arraySize;
  q[IV] = Q[offset];  offset += arraySize;
  q[IW] = Q[offset];  offset += arraySize;
  q[IA] = Q[offset];  offset += arraySize;
  q[IB] = Q[offset];  offset += arraySize;
  q[IC] = Q[offset];

} // getPrimitiveVector


/**
 * get face-centered mag Field at some location.
 *
 * \param[in]  U          Conserved variable raw array
 * \param[in]  arraySize  Domain size 
 * \param[in]  elemOffset offset to current location
 * \param[out] bf         returned magnetic field
 */
inline __DEVICE__
void getMagField(real_t*  U, 
		 int      arraySize, 
		 int      elemOffset, 
		 real_t (&bf)[3])
{

  int offset = elemOffset + 5 * arraySize;
  bf[IX]  = U[offset];  offset += arraySize;
  bf[IY]  = U[offset];  offset += arraySize;
  bf[IZ]  = U[offset];

} // getMagField

#endif /*CONSTOPRIM_H_*/
