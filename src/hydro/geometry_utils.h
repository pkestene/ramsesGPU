/**
 * \file geometry_utils.h
 * \brief Small geometry related utilities common to CPU / GPU code.
 *
 * This utility function (compute_ds_dv) is directly
 * adapted from Fortran original code found in DUMSES.
 *
 * \date 20 January 2012
 * \author Pierre Kestener.
 *
 * $Id: geometry_utils.h 1784 2012-02-21 10:34:58Z pkestene $
 */
#ifndef GEOMETRY_UTILS_H_
#define GEOMETRY_UTILS_H_

#include "constants.h"

/**
 * Compute surface elements and volume for a given cell in a given geometry.
 *
 * Here we only deal with cylindrical and spherical geometry. The cartesian case is
 * handle directly in the calling code (since dv and ds do not depend on coordinate
 * i,j,k it would be inefficient to do it here).
 *
 * \param[out] ds vector of surface elements
 * \param[out] dv cell volume
 * \param[in]  i  x coordinate
 * \param[in]  j  y coordinate
 * \param[in]  k  z coordinate
 *
 * \tparam NDIM 2 or 3
 * \tparam geometry GEO_CYLINDRICAL or GEO_SPHERICAL
 */
template<DimensionType NDIM, GeometryType geometry>
__DEVICE__
void compute_ds_dv(real_t (&ds)[NDIM], real_t &dv, int i, int j, int k, int ghostWidth)
{

  //int &geometry = ::gParams.geometry;
  real_t &dx    = ::gParams.dx;
  real_t &dy    = ::gParams.dy;
  real_t &dz    = ::gParams.dz;

  if (NDIM == TWO_D) {

    /*if (geometry == GEO_CARTESIAN) {
      dv     = dx*dy;
      ds[IX] = dy;
      ds[IY] = dx;
      }*/

    if (geometry == GEO_CYLINDRICAL) {
      real_t xPos = ::gParams.xMin + dx/2 + (i-ghostWidth)*dx;
      dv = dx*xPos*dy;
      ds[IX] = (xPos+dx/2)*dy;
      ds[IY] = dx;
    }

  } else { // THREE_D

    /*if (geometry == GEO_CARTESIAN) {
      dv     = dx*dy*dz;
      ds[IX] = dy*dz;
      ds[IY] = dx*dz;
      ds[IZ] = dx*dy;
      }*/

    if (geometry == GEO_CYLINDRICAL) {
      real_t xPos = ::gParams.xMin + dx/2 + (i-ghostWidth)*dx;
      dv = dx*xPos*dy;
      ds[IX] = (xPos+dx/2)*dy*dz;
      ds[IY] = dx*dz;
      ds[IZ] = xPos*dx*dy;
    }

    if (geometry == GEO_SPHERICAL) {
      real_t xPos = ::gParams.xMin + dx/2 + (i-ghostWidth)*dx;
      real_t yPos = ::gParams.yMin + dy/2 + (j-ghostWidth)*dy;
      real_t zPos = ::gParams.zMin + dz/2 + (k-ghostWidth)*dz;
      dv = dx*xPos*dy*xPos*sin(yPos)*dz;
      ds[IX] = (xPos+dx/2)*dy*dz;
      ds[IY] = (xPos+dx/2)*dx*dz;
      ds[IZ] = xPos*sin(zPos)*dx*dy;
    }

  } // end THREE_D
  

} // compute_ds_dv


#endif // GEOMETRY_UTILS_H_
