/*
 * Copyright CEA / Maison de la Simulation
 * Contributors: Pierre Kestener, Sebastien Fromang (May 22, 2012)
 *
 * This software is governed by the CeCILL license under French law and
 * abiding by the rules of distribution of free software.  You can  use, 
 * modify and/ or redistribute the software under the terms of the CeCILL
 * license as circulated by CEA, CNRS and INRIA at the following URL
 * "http://www.cecill.info". 
 *
 * The fact that you are presently reading this means that you have had
 * knowledge of the CeCILL license and that you accept its terms.
 */

/**
 * \file initHydro.cpp
 * \brief Provides a routine to pre-compute all 2D Riemann problems configurations.
 *
 * \author P. Kestener
 *
 * $Id: initHydro.cpp 2108 2012-05-23 12:07:21Z pkestene $
 */
#include "initHydro.h"

void initRiemannConfig2d(RiemannConfig2d (&conf)[NB_RIEMANN_CONFIG])
{
  // Config 1
  conf[0].pvar[0].rho = 1.0f;
  conf[0].pvar[0].u   = 0.0f;
  conf[0].pvar[0].v   = 0.0f;
  conf[0].pvar[0].p   = 1.0f;

  conf[0].pvar[1].rho = 0.5197f;
  conf[0].pvar[1].u   =-0.7259f;
  conf[0].pvar[1].v   = 0.0f;
  conf[0].pvar[1].p   = 0.4f;

  conf[0].pvar[2].rho = 0.1072f;
  conf[0].pvar[2].u   =-0.7259f;
  conf[0].pvar[2].v   =-1.4045f;
  conf[0].pvar[2].p   = 0.0439f;

  conf[0].pvar[3].rho = 0.2579f;
  conf[0].pvar[3].u   = 0.0f;
  conf[0].pvar[3].v   =-1.4045f;
  conf[0].pvar[3].p   = 0.15f;

  // Config 2
  conf[1].pvar[0].rho = 1.0f;
  conf[1].pvar[0].u   = 0.0f;
  conf[1].pvar[0].v   = 0.0f;
  conf[1].pvar[0].p   = 1.0f;

  conf[1].pvar[1].rho = 0.5197f;
  conf[1].pvar[1].u   =-0.7259f;
  conf[1].pvar[1].v   = 0.0f;
  conf[1].pvar[1].p   = 0.4f;

  conf[1].pvar[2].rho = 1.0f;
  conf[1].pvar[2].u   =-0.7259f;
  conf[1].pvar[2].v   =-0.7259f;
  conf[1].pvar[2].p   = 1.0f;

  conf[1].pvar[3].rho = 0.5197f;
  conf[1].pvar[3].u   = 0.0f;
  conf[1].pvar[3].v   =-0.7259f;
  conf[1].pvar[3].p   = 0.4f;

  // Config 3
  conf[2].pvar[0].rho = 1.5f;
  conf[2].pvar[0].u   = 0.0f;
  conf[2].pvar[0].v   = 0.0f;
  conf[2].pvar[0].p   = 1.5f;

  conf[2].pvar[1].rho = 0.5323f;
  conf[2].pvar[1].u   = 1.206f;
  conf[2].pvar[1].v   = 0.0f;
  conf[2].pvar[1].p   = 0.3f;

  conf[2].pvar[2].rho = 0.138f;
  conf[2].pvar[2].u   = 1.206f;
  conf[2].pvar[2].v   = 1.206f;
  conf[2].pvar[2].p   = 0.029f;

  conf[2].pvar[3].rho = 0.5323f;
  conf[2].pvar[3].u   = 0.0f;
  conf[2].pvar[3].v   = 1.206f;
  conf[2].pvar[3].p   = 0.3f;

  // Config 4
  conf[3].pvar[0].rho = 1.1f;
  conf[3].pvar[0].u   = 0.0f;
  conf[3].pvar[0].v   = 0.0f;
  conf[3].pvar[0].p   = 1.1f;

  conf[3].pvar[1].rho = 0.5065f;
  conf[3].pvar[1].u   = 0.8939f;
  conf[3].pvar[1].v   = 0.0f;
  conf[3].pvar[1].p   = 0.35f;

  conf[3].pvar[2].rho = 1.1f;
  conf[3].pvar[2].u   = 0.8939f;
  conf[3].pvar[2].v   = 0.8939f;
  conf[3].pvar[2].p   = 1.1f;

  conf[3].pvar[3].rho = 0.5065f;
  conf[3].pvar[3].u   = 0.0f;
  conf[3].pvar[3].v   = 0.8939f;
  conf[3].pvar[3].p   = 0.35f;

  // Config 5
  conf[4].pvar[0].rho = 1.0f;
  conf[4].pvar[0].u   =-0.75f;
  conf[4].pvar[0].v   =-0.5f;
  conf[4].pvar[0].p   = 1.0f;

  conf[4].pvar[1].rho = 2.0f;
  conf[4].pvar[1].u   =-0.75f;
  conf[4].pvar[1].v   = 0.5f;
  conf[4].pvar[1].p   = 1.0f;

  conf[4].pvar[2].rho = 1.0f;
  conf[4].pvar[2].u   = 0.75f;
  conf[4].pvar[2].v   = 0.5f;
  conf[4].pvar[2].p   = 1.0f;

  conf[4].pvar[3].rho = 3.0f;
  conf[4].pvar[3].u   = 0.75f;
  conf[4].pvar[3].v   =-0.5f;
  conf[4].pvar[3].p   = 1.0f;

  // Config 6
  conf[5].pvar[0].rho = 1.0f;
  conf[5].pvar[0].u   = 0.75f;
  conf[5].pvar[0].v   =-0.5f;
  conf[5].pvar[0].p   = 1.0f;

  conf[5].pvar[1].rho = 2.0f;
  conf[5].pvar[1].u   = 0.75f;
  conf[5].pvar[1].v   = 0.5f;
  conf[5].pvar[1].p   = 0.5f;

  conf[5].pvar[2].rho = 1.0f;
  conf[5].pvar[2].u   =-0.75f;
  conf[5].pvar[2].v   = 0.5f;
  conf[5].pvar[2].p   = 1.0f;

  conf[5].pvar[3].rho = 3.0f;
  conf[5].pvar[3].u   =-0.75f;
  conf[5].pvar[3].v   =-0.5f;
  conf[5].pvar[3].p   = 1.0f;

  // Config 7
  conf[6].pvar[0].rho = 1.0f;
  conf[6].pvar[0].u   = 0.1f;
  conf[6].pvar[0].v   = 0.1f;
  conf[6].pvar[0].p   = 1.0f;

  conf[6].pvar[1].rho = 0.5197f;
  conf[6].pvar[1].u   =-0.6259f;
  conf[6].pvar[1].v   = 0.1f;
  conf[6].pvar[1].p   = 0.4f;

  conf[6].pvar[2].rho = 0.8f;
  conf[6].pvar[2].u   = 0.1f;
  conf[6].pvar[2].v   = 0.1f;
  conf[6].pvar[2].p   = 0.4f;

  conf[6].pvar[3].rho = 0.5197f;
  conf[6].pvar[3].u   = 0.1f;
  conf[6].pvar[3].v   =-0.6259f;
  conf[6].pvar[3].p   = 0.4f;

  // Config 8
  conf[7].pvar[0].rho = 0.5197f;
  conf[7].pvar[0].u   = 0.1f;
  conf[7].pvar[0].v   = 0.1f;
  conf[7].pvar[0].p   = 0.4f;

  conf[7].pvar[1].rho = 1.0f;
  conf[7].pvar[1].u   =-0.6259f;
  conf[7].pvar[1].v   = 0.1f;
  conf[7].pvar[1].p   = 1.0f;

  conf[7].pvar[2].rho = 0.8f;
  conf[7].pvar[2].u   = 0.1f;
  conf[7].pvar[2].v   = 0.1f;
  conf[7].pvar[2].p   = 1.0f;

  conf[7].pvar[3].rho = 1.0f;
  conf[7].pvar[3].u   = 0.1f;
  conf[7].pvar[3].v   =-0.6259f;
  conf[7].pvar[3].p   = 1.0f;

  // Config 9
  conf[8].pvar[0].rho = 1.0f;
  conf[8].pvar[0].u   = 0.0f;
  conf[8].pvar[0].v   = 0.3f;
  conf[8].pvar[0].p   = 1.0f;

  conf[8].pvar[1].rho = 2.0f;
  conf[8].pvar[1].u   = 0.0f;
  conf[8].pvar[1].v   =-0.3f;
  conf[8].pvar[1].p   = 1.0f;

  conf[8].pvar[2].rho = 1.039f;
  conf[8].pvar[2].u   = 0.0f;
  conf[8].pvar[2].v   =-0.8133f;
  conf[8].pvar[2].p   = 0.4f;

  conf[8].pvar[3].rho = 0.5197f;
  conf[8].pvar[3].u   = 0.0f;
  conf[8].pvar[3].v   =-0.4259f;
  conf[8].pvar[3].p   = 0.4f;

  // Config 10
  conf[9].pvar[0].rho = 1.0f;
  conf[9].pvar[0].u   = 0.0f;
  conf[9].pvar[0].v   = 0.4297f;
  conf[9].pvar[0].p   = 1.0f;

  conf[9].pvar[1].rho = 0.5f;
  conf[9].pvar[1].u   = 0.0f;
  conf[9].pvar[1].v   = 0.6076f;
  conf[9].pvar[1].p   = 1.0f;

  conf[9].pvar[2].rho = 0.2281f;
  conf[9].pvar[2].u   = 0.0f;
  conf[9].pvar[2].v   =-0.6076f;
  conf[9].pvar[2].p   = 0.3333f;

  conf[9].pvar[3].rho = 0.4562f;
  conf[9].pvar[3].u   = 0.0f;
  conf[9].pvar[3].v   =-0.4259f;
  conf[9].pvar[3].p   = 0.3333f;

  // Config 11
  conf[10].pvar[0].rho = 1.0f;
  conf[10].pvar[0].u   = 0.1f;
  conf[10].pvar[0].v   = 0.0f;
  conf[10].pvar[0].p   = 1.0f;

  conf[10].pvar[1].rho = 0.5313f;
  conf[10].pvar[1].u   = 0.8276f;
  conf[10].pvar[1].v   = 0.0f;
  conf[10].pvar[1].p   = 0.4f;

  conf[10].pvar[2].rho = 0.8f;
  conf[10].pvar[2].u   = 0.1f;
  conf[10].pvar[2].v   = 0.0f;
  conf[10].pvar[2].p   = 0.4f;

  conf[10].pvar[3].rho = 0.5313f;
  conf[10].pvar[3].u   = 0.1f;
  conf[10].pvar[3].v   = 0.7276f;
  conf[10].pvar[3].p   = 0.4f;

  // Config 12
  conf[11].pvar[0].rho = 0.5313f;
  conf[11].pvar[0].u   = 0.0f;
  conf[11].pvar[0].v   = 0.0f;
  conf[11].pvar[0].p   = 0.4f;

  conf[11].pvar[1].rho = 1.0f;
  conf[11].pvar[1].u   = 0.7276f;
  conf[11].pvar[1].v   = 0.0f;
  conf[11].pvar[1].p   = 1.0f;

  conf[11].pvar[2].rho = 0.8f;
  conf[11].pvar[2].u   = 0.0f;
  conf[11].pvar[2].v   = 0.0f;
  conf[11].pvar[2].p   = 1.0f;

  conf[11].pvar[3].rho = 1.0f;
  conf[11].pvar[3].u   = 0.0f;
  conf[11].pvar[3].v   = 0.7276f;
  conf[11].pvar[3].p   = 1.0f;

  // Config 13
  conf[12].pvar[0].rho = 1.0f;
  conf[12].pvar[0].u   = 0.0f;
  conf[12].pvar[0].v   =-0.3f;
  conf[12].pvar[0].p   = 1.0f;

  conf[12].pvar[1].rho = 2.0f;
  conf[12].pvar[1].u   = 0.0f;
  conf[12].pvar[1].v   = 0.3f;
  conf[12].pvar[1].p   = 1.0f;

  conf[12].pvar[2].rho = 1.0625f;
  conf[12].pvar[2].u   = 0.0f;
  conf[12].pvar[2].v   = 0.8145f;
  conf[12].pvar[2].p   = 0.4f;

  conf[12].pvar[3].rho = 0.5313f;
  conf[12].pvar[3].u   = 0.0f;
  conf[12].pvar[3].v   = 0.4276f;
  conf[12].pvar[3].p   = 0.4f;

  // Config 14
  conf[13].pvar[0].rho = 2.0f;
  conf[13].pvar[0].u   = 0.0f;
  conf[13].pvar[0].v   =-0.5606f;
  conf[13].pvar[0].p   = 8.0f;

  conf[13].pvar[1].rho = 1.0f;
  conf[13].pvar[1].u   = 0.0f;
  conf[13].pvar[1].v   =-1.2172f;
  conf[13].pvar[1].p   = 8.0f;

  conf[13].pvar[2].rho = 0.4736f;
  conf[13].pvar[2].u   = 0.0f;
  conf[13].pvar[2].v   = 1.2172f;
  conf[13].pvar[2].p   = 2.6667f;

  conf[13].pvar[3].rho = 0.9474f;
  conf[13].pvar[3].u   = 0.0f;
  conf[13].pvar[3].v   = 1.1606f;
  conf[13].pvar[3].p   = 2.6667f;

  // Config 15
  conf[14].pvar[0].rho = 1.0f;
  conf[14].pvar[0].u   = 0.1f;
  conf[14].pvar[0].v   =-0.3f;
  conf[14].pvar[0].p   = 1.0f;

  conf[14].pvar[1].rho = 0.5197f;
  conf[14].pvar[1].u   =-0.6259f;
  conf[14].pvar[1].v   =-0.3f;
  conf[14].pvar[1].p   = 0.4f;

  conf[14].pvar[2].rho = 0.8f;
  conf[14].pvar[2].u   = 0.1f;
  conf[14].pvar[2].v   =-0.3f;
  conf[14].pvar[2].p   = 0.4f;

  conf[14].pvar[3].rho = 0.5313f;
  conf[14].pvar[3].u   = 0.1f;
  conf[14].pvar[3].v   = 0.4276f;
  conf[14].pvar[3].p   = 0.4f;

  // Config 16
  conf[15].pvar[0].rho = 0.5313f;
  conf[15].pvar[0].u   = 0.1f;
  conf[15].pvar[0].v   = 0.1f;
  conf[15].pvar[0].p   = 0.4f;

  conf[15].pvar[1].rho = 1.0222f;
  conf[15].pvar[1].u   =-0.6179f;
  conf[15].pvar[1].v   = 0.1f;
  conf[15].pvar[1].p   = 1.0f;

  conf[15].pvar[2].rho = 0.8f;
  conf[15].pvar[2].u   = 0.1f;
  conf[15].pvar[2].v   = 0.1f;
  conf[15].pvar[2].p   = 1.0f;

  conf[15].pvar[3].rho = 1.0f;
  conf[15].pvar[3].u   = 0.1f;
  conf[15].pvar[3].v   = 0.8276f;
  conf[15].pvar[3].p   = 1.0f;

  // Config 17
  conf[16].pvar[0].rho = 1.0f;
  conf[16].pvar[0].u   = 0.0f;
  conf[16].pvar[0].v   =-0.4f;
  conf[16].pvar[0].p   = 1.0f;

  conf[16].pvar[1].rho = 2.0f;
  conf[16].pvar[1].u   = 0.0f;
  conf[16].pvar[1].v   =-0.3f;
  conf[16].pvar[1].p   = 1.0f;

  conf[16].pvar[2].rho = 1.0625f;
  conf[16].pvar[2].u   = 0.0f;
  conf[16].pvar[2].v   = 0.2145f;
  conf[16].pvar[2].p   = 0.4f;

  conf[16].pvar[3].rho = 0.5197f;
  conf[16].pvar[3].u   = 0.0f;
  conf[16].pvar[3].v   =-1.1259f;
  conf[16].pvar[3].p   = 0.4f;

  // Config 18
  conf[17].pvar[0].rho = 1.0f;
  conf[17].pvar[0].u   = 0.0f;
  conf[17].pvar[0].v   = 1.0f;
  conf[17].pvar[0].p   = 1.0f;

  conf[17].pvar[1].rho = 2.0f;
  conf[17].pvar[1].u   = 0.0f;
  conf[17].pvar[1].v   =-0.3f;
  conf[17].pvar[1].p   = 1.0f;

  conf[17].pvar[2].rho = 1.0625f;
  conf[17].pvar[2].u   = 0.0f;
  conf[17].pvar[2].v   = 0.2145f;
  conf[17].pvar[2].p   = 0.4f;

  conf[17].pvar[3].rho = 0.5197f;
  conf[17].pvar[3].u   = 0.0f;
  conf[17].pvar[3].v   = 0.2741f;
  conf[17].pvar[3].p   = 0.4f;

  // Config 19
  conf[18].pvar[0].rho = 1.0f;
  conf[18].pvar[0].u   = 0.0f;
  conf[18].pvar[0].v   = 0.3f;
  conf[18].pvar[0].p   = 1.0f;

  conf[18].pvar[1].rho = 2.0f;
  conf[18].pvar[1].u   = 0.0f;
  conf[18].pvar[1].v   =-0.3f;
  conf[18].pvar[1].p   = 1.0f;

  conf[18].pvar[2].rho = 1.0625f;
  conf[18].pvar[2].u   = 0.0f;
  conf[18].pvar[2].v   = 0.2145f;
  conf[18].pvar[2].p   = 0.4f;

  conf[18].pvar[3].rho = 0.5197f;
  conf[18].pvar[3].u   = 0.0f;
  conf[18].pvar[3].v   =-0.4259f;
  conf[18].pvar[3].p   = 0.4f;

}
