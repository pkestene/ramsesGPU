/**
 * \file initHydro.h
 * \brief Implement initialization routine to solve a 2D Riemann
 * problem.
 *
 * In the 2D case, there are 19 different possible configurations (see
 * article by Lax and Liu, "Solution of two-dimensional riemann
 * problems of gas dynamics by positive schemes",SIAM journal on
 * scientific computing, 1998, vol. 19, no2, pp. 319-340).
 *
 * \author P. Kestener
 *
 * $Id: initHydro.h 1784 2012-02-21 10:34:58Z pkestene $
 */
#ifndef INIT_HYDRO_H_
#define INIT_HYDRO_H_

// number 2D Riemann problems configuration
#define NB_RIEMANN_CONFIG 19

/**
 * \struct PrimitiveEuler initHydro.h
 * \brief a simple structure gather hydro fields in a simulation run.
 */
struct PrimitiveEuler {
  
  PrimitiveEuler() : rho(0.0f), u(0.0f), v(0.0f), p(0.0f) {};
  float rho;
  float u;
  float v;
  float p;
};

/**
 * \struct RiemannConfig2d initHydro.h
 * \brief A very simple structure to store the parameters of all 19 possible
 * 2D Riemann problems. 
 * 
 * See article: Lax and Liu, "Solution of two-dimensional riemann
 * problems of gas dynamics by positive schemes",SIAM journal on
 * scientific computing, 1998, vol. 19, no2, pp. 319-340
 */
struct RiemannConfig2d {

  RiemannConfig2d() {};
  PrimitiveEuler pvar[4];

};

void initRiemannConfig2d(RiemannConfig2d (&conf)[NB_RIEMANN_CONFIG]);

#endif // INIT_HYDRO_H_
