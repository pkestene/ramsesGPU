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
 * \file zSlabInfo.h
 * \brief Defines structure holding info about a given z-slab.
 *
 * \date 13 Sept 2012
 * \author P. Kestener
 *
 * $Id: zSlabInfo.h 2395 2012-09-14 12:45:17Z pkestene $
 */
#ifndef ZSLAB_INFO_H_
#define ZSLAB_INFO_H_

/**
 * \struct ZslabInfo
 */
typedef struct {

  int zSlabId;     //!< slab Id
  int zSlabNb;     //!< total number of slabs
  int zSlabWidthG; //!< slab width along z, ghost included
  int kStart;      //!< index of the first z-plane in the slab
  int kStop;       //!< index of the next z-plane above the slab
  int ksizeSlab;   //!< effective width of the slab (the very slab might be smaller).

} ZslabInfo;

#endif // ZSLAB_INFO_H_
