/**
 * \file pnetcdf_io.h
 *
 * Parallel NetCDF IO.
 *
 * \author P. Kestener
 * \date 20/06/2013
 *
 * $Id: pnetcdf_io.h 3369 2014-04-14 15:52:22Z pkestene $
 */

#ifndef PNETCDF_IO_H_
#define PNETCDF_IO_H_

#include <ConfigMap.h>
#include <Arrays.h>
using hydroSimu::HostArray;

/**
 * Read netcdf file using Parallel-NetCDF, removes ghosts.
 * Memory allocation for HostArray is done outside.
 *
 */
void read_pnetcdf(const std::string &filename,
		  int                iVar,
		  ConfigMap         &configMap,
		  HostArray<double> &localData);

/**
 * Write netcdf file using Parallel-NetCDF, with/without ghosts.
 *
 * NetCDF file creation supports:
 * - CDF-2 (using creation mode NC_64BIT_OFFSET)
 * - CDF-5 (using creation mode NC_64BIT_DATA)
 *
 * Write variable rho, rho_vx, rho_vy, rho_vz from a sub-domain inside NetCDF file.
 *
 * \param[in]    filename (input data)
 * \param[inout] localData A reference to a HostArray (data to write)
 *
 */
void write_pnetcdf(const std::string &filename,
		   HostArray<double> &localData,
		   ConfigMap         &configMap);

#endif /* PNETCDF_IO_H_ */
