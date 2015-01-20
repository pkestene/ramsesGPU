/**
 * \file base_type.h
 * \brief Defines some usefull base types.
 *
 * The idea is the replace declaration of type real_t array (for hydro state) by
 * a structure.
 *
 * \date March 14, 2012
 * \author Pierre Kestener
 *
 * $Id: base_type.h 2395 2012-09-14 12:45:17Z pkestene $
 */
#ifndef BASE_TYPE_H_
#define BASE_TYPE_H_

#include "real_type.h"

/**
 * a simple structure designed to replace a real_t array in pure hydro computation.
 *
 * On GPU declaring an array results in register spilling (since register are not 
 * indexable.
 */
struct qStateHydro
{
  real_t D;
  real_t P;
  real_t U;
  real_t V;
  real_t W;

  __DEVICE__ void reset() {
    D = ZERO_F;
    P = ZERO_F;
    U = ZERO_F;
    V = ZERO_F;
    W = ZERO_F;
  };

}; // qStateHydro

/**
 * a simple structure designed to replace a real_t array in pure hydro computation.
 */
struct qStateMHD
{
  real_t D;
  real_t P;
  real_t U;
  real_t V;
  real_t W;
  real_t A;
  real_t B;
  real_t C;

  __DEVICE__ void reset() {
    D = ZERO_F;
    P = ZERO_F;
    U = ZERO_F;
    V = ZERO_F;
    W = ZERO_F;
    A = ZERO_F;
    B = ZERO_F;
    C = ZERO_F;
  };

}; // qStateMHD


#endif // BASE_TYPE_H_
