/**
 * \file main.h
 * \brief Doxygen main page Documentation.
 * \author P. Kestener
 *
 * \defgroup test
 *
 * \mainpage
 *
 * \section Purpose Purpose
 *
 * RAMSES-GPU is a software package providing a C++/Cuda
 * implementation of several 2nd order numerical schemes for solving the Euler
 * equations (2d and 3D) on heterogenous distributed architectures
 * (CPU or GPU + MPI) and also MHD with shearing box.
 *
 * RAMSES-GPU is developped by Maison de la Simulation (http://www.maisondelasimulation.fr) and CEA/Sap (http://irfu.cea.fr/Sap/en/index.php)
 *
 * RAMSES-GPU is governed by the CeCILL  license http://www.cecill.info
 * The fact that you are presently reading this means that you have had
 * knowledge of the CeCILL license and that you accept its terms.
 *
 * RAMSES-GPU sources can be downloaded at http://www.maisondelasimulation.fr/projects/RAMSES-GPU/html/download.html
 *
 * \htmlinclude simu.html
 *
 * \section Features Features
 *
 * Here is a table sumarizing software capabilities for each numerical scheme:
 * <table>
 * <tr><td>Numerical Scheme</td> <td>CPU</td> <td>GPU</td> <td>CPU+MPI</td> <td>GPU+MPI</td> </tr>
 * <tr><td>Hydro: MUSCL split</td> <td>2D and 3D</td> <td>2D and
 * 3D</td> <td>2D and 3D</td> <td>2D and 3D</td></tr>
 * <tr><td>Hydro: MUSCL unsplit</td> <td>2D and 3D</td> <td>2D and
 * 3D</td> <td>2D and 3D</td> <td>2D and 3D</td></tr>
 * <tr><td>Hydro: Kurganov</td> <td>2D only</td> <td>2D only</td> <td>No</td> <td>No</td> </tr>
 * <tr><td>Hydro: Relaxing TVD</td> <td>2D only</td> <td>2D only</td>
 * <td>No</td> <td>No</td></tr>
 * <tr><td>MHD: MUSCL unsplit, CT</td> <td>2D/3D</td> <td>2D/3D</td>
 * <td>2D/3D</td> <td>2D/3D</td></tr> 
 * </table>
 *
 * MHD simulations can also use shearing-box border conditions (used mainly for Magneto-Rotational Instability studies).
 *
 * \subsection About MHD implementations
 * Have a look at class MHDRunGonunov documentation. 
 *
 * \subsection Simulation Simulation output file formats
 * \li \c HDF5 with a XDMF header (usefull to make paraview read HDF5
 * data files)
 * \li \c VTK Image Data, a XML configurable format. If VTK library is
 * available, the code the VTK API, if not the code hand-write the VTK
 * file format using the ASCII or raw binary variant (base64 encoding
 * and compression are not implemented). 
 *
 * The MPI version of the code can use parallel HDF5 (one single file,
 * each MPI process only writes a subset of the whole data using
 * hyperslab and PHDF5 API, see) or parallel VTK (one file per MPI
 * process, ASCII or raw binary, no compression, no base64
 * compression).
 *
 * \subsection Parameters Simulation parameters
 * The code is configurable at run-time by giving a parameter file
 * (using the INI file format). There are many example configuration
 * parameter files in the \c data sub-directory.
 *
 * A parameter file is organized into sections :
 * <UL>
 * <LI> section \c run : 
 *    <UL>
 *        <LI>\c nstemax total number of time step</LI>
 *        <LI>\c tend maximum end time</LI>
 *        <LI>\c noutput number of time steps between 2 outputs </LI>
 *    </UL>
 * </LI>
 * <LI> section \c mesh : local geometry parameters
 *    <UL>
 *        <LI>\c nx, \c ny, and \c nz : sizes of the cartesian mesh
 * grid (number of cells for one process)</LI>
 *        <LI>\c boundary_xmin, etc ... : an integer characterizing
 * the border condition for each border; can be Dirichlet, Neumann or periodic</LI>
 *    </UL>
 * </LI>
 * <LI> section \c mpi [optional] : global geometry parameters
 *    <UL>
 *        <LI>\c mx, \c my, and \c mz : sizes of the cartesian mesh
 * (number of MPI processes)</LI>
 *    </UL>
 * </LI>
 * <LI> section \c hydro :
 *    <UL>
 *        <LI> \c problem : name of problem (used for defining init conditions)</LI>
 *        <LI> \c cfl : Courant-Friedrichs-Lewy number </LI>
 *        <LI> \c niter_riemann : used in the \c approx Riemann solver </LI>
 *        <LI> \c iorder : numerical scheme order (use 2 for enabling
 * slope computations</LI>
 *        <LI> \c slope_type : used in slope computations </LI>
 *        <LI> \c scheme : switch between muscl, plmde or collela (see
 * RAMSES doc for more information)
 * </LI>
 *        <LI> \c traceVersion : switch between the different trace
 * computation GPU implementations (0 means no trace, 1 or 2 means
 * trace enabled). </LI>
 *        <LI> \c riemannSolver : name of the Riemann solver; approx
 * or HLLC for hydro; HLLD for MHD </LI>
 *        <LI> \c smallr : small density cut-off </LI>
 *        <LI> \c smallc : small speed of sound cut-off </LI>
 *    </UL>
 * </LI>
 * <LI> section \c output :
 *    <UL>
 *        <LI> \c outputDir : name of the output directory </LI>
 *        <LI> \c outputPrefix : prefix used in output file names</LI>
 *        <LI> \c outputVtk : enable/disable VTK output</LI>
 *        <LI> \c outputVtkHandWritten : enable/disable the
 * hand-written version (if VTK library is not available, always
 * fallback on the hand-written version)</LI>
 *        <LI> \c outputVtkCompression : enable/disable compression</LI>
 *        <LI> \c outputVtkAscii : enable/disable ASCII output</LI>
 *        <LI> \c outputVtkBase64Encoding : enable/disable base64 encoding</LI>
 *        <LI> \c outputHdf5 : enable/disable HDF5 </LI>
 *        <LI> \c outputXsm : use XSM file format (serial version only)</LI>
 *        <LI> \c outputPng : use image PNG format (serial version only, 2D only)</LI>
 *    </UL>
 * </LI>
 * </UL>
 *
 * \section License License
 *
 * Copyright CEA / Maison de la Simulation\n
 * Contributors: Pierre Kestener, Sebastien Fromang (May 22, 2012)
 *
 * pierre.kestener@cea.fr
 *
 * This software is a computer program whose purpose is to provide GPU implementations of some finite volume numerical schemes used to perform hydrodynamics and MHD flow simulations.
 *
 * This software is governed by the CeCILL  license under French law and
 * abiding by the rules of distribution of free software.  You can  use, 
 * modify and/ or redistribute the software under the terms of the CeCILL
 * license as circulated by CEA, CNRS and INRIA at the following URL
 * "http://www.cecill.info". 
 *
 * As a counterpart to the access to the source code and  rights to copy,
 * modify and redistribute granted by the license, users are provided only
 * with a limited warranty  and the software's author,  the holder of the
 * economic rights,  and the successive licensors  have only  limited
 *liability. 
 *
 * In this respect, the user's attention is drawn to the risks associated
 * with loading,  using,  modifying and/or developing or reproducing the
 * software by the user in light of its specific status of free software,
 * that may mean  that it is complicated to manipulate,  and  that  also
 * therefore means  that it is reserved for developers  and  experienced
 * professionals having in-depth computer knowledge. Users are therefore
 * encouraged to load and test the software's suitability as regards their
 * requirements in conditions enabling the security of their systems and/or 
 * data to be ensured and,  more generally, to use and operate it in the 
 * same conditions as regards security. 
 *
 * The fact that you are presently reading this means that you have had
 * knowledge of the CeCILL license and that you accept its terms.
 *
 *
 * \section Get-the-code Get the code
 *
 * \verbatim svn co https://dsm-trac.cea.fr/svn/coast/gpu_tryouts/HybridHydro/trunk trunk \endverbatim
 *
 * \section Requirements Requirements
 *
 * You need a relatively decent/recent CUDA-capable graphics board
 * (see page <a
 * href="http://en.wikipedia.org/wiki/Comparison_of_Nvidia_graphics_processing_units">list
 * of CUDA-capable device</a>) with hardware capability 1.3.
 * 
 * You also need to have the Nvidia toolkit (NVCC compiler) installed
 * to build the GPU version of the code. If it is not found, only the
 * CPU version will be build.
 *
 * \section Compile Compile / Build 
 *
 * \li \c Preparation: Execute once shell script \e autogen.sh
 * (autotools build system) 
 * \li \c configure: Execute the configure script inside a build
 * directory (that could be \e $(SRCDIR)/build for example) \code 
 configure --with-cuda=/usr/local/cuda30 \endcode
 * Type \code configure --help \endcode to see other option
 * \li \c build: Type \code make \endcode this creates several
 * executables (\e euler_cpu , \e euler_gpu) in the src sub directory
 * \li \c example \c of \c 2d \c run \c on \c CPU: Type \code ./euler_cpu --param jet.ini \endcode to run
 * \li \c example \c of \c 2d \c run \c on \c GPU: Type \code ./euler_gpu --param jet.ini \endcode
 *
 * \li \c example \c of \c 3d \c run \c on \c GPU: \code ./euler_gpu --param jet3d.ini \endcode
 * 
 * \li \c input \c parameters: see example file in directory \e data
 * (using the minimalist parameter file parser  <a href="http://getpot.sourceforge.net/">GetPot</a>)
 * \li \c output \c files: The default ouput file format is VTK (image
 * data, extension \e .vti )
 *
 * \li \c Qt \c GUI \c of \c 2D \c simulations: You can see \b live
 * \b results of computations using the Qt-based GUI: 
 * \code euler2d_gpu_qt --param jet.ini\endcode 
 *
 * \section Build_tgcc Build at GENCI/TGCC (Curie)
 *
 * The default build is to use the Intel icc/icpc compiler.
 *
 * - \c modules: cuda/4.1, phdf5
 *
 * - \c configure \c line \c at \c CCRT (with GPU, MPI and double precision enabled):
 *   - NVCCFLAGS="-gencode=arch=compute_20,code=sm_20 " ../trunk/configure --disable-shared --with-cuda=/usr/local/cuda-4.1 --with-boost-mpi=no --disable-qtgui --enable-mpi --enable-timing --enable-double CC=icc CXX=icpc
 *
 * - \c other parameters: you can env variable MAX_REG_COUNT_SINGLE or MAX_REG_COUNT_DOUBLE to increase the maximun cuda register at compile time
 *
 * - \c submission script for a GPU+MPI job with 8 MPI processes (each of which accessing 1 GPU):
 *    \code
 *     #!/bin/bash 
 *     #MSUB -r MRI_MPI_GPU               # Request name 
 *     #MSUB -n 8                         # Total number of tasks to use 
 *     #MSUB -N 4                         # Total number of nodes
 *     #MSUB -T 14400                     # Elapsed time limit in seconds 
 *     #MSUB -o mri3d_gpu_mpi_%I.out      # Standard output. %I is the job id 
 *     #MSUB -e mri3d_gpu_mpi_%I.err      # Error output. %I is the job id 
 *     #MSUB -q hybrid                    # Hybrid partition of GPU nodes
 *     #MSUB -A gen2231                   # Project ID
 *
 *     set -x 
 *     cd ${BRIDGE_MSUB_PWD} 
 *     module load cuda/4.1
 *     module load phdf5
 *
 *     ccc_mprun ./euler_gpu_mpi --param ./mhd_mri_3d_gpu_mpi.ini
 *    \endcode
 * Submit your job with
 *    ccc_msub job_multiGPU.sh
 *
 * \section Build_ccrt Build at CCRT (Titane)
 *
 * The default build is to use the Intel icc/icpc compiler.
 *
 * - \c modules: before trying to compile/build, you need to put the
 *    following modules in you environment:
 *    - module add cuda/3.2
 *    - module add autotools/09.10.28
 *    - module add hdf5/1.8.2_bullxmpi
 *
 * - \c svn \c and \c M4: if you use the SVN sources, you need to execute first the
 *    bash script autogen.sh and since the autotools are not fully
 *    installed (some extra M4 macro are missing or not installed in
 *    the regular place), you need to add pkg.m4 and acx_mpi.m4 in
 *    the local m4 directory.
 *
 * - \c configure \c line \c at \c CCRT:
 *   - ../build_trunk/configure --disable-shared --with-cuda=/applications/cuda-3.2/ --with-boost-mpi=no --disable-qtgui --enable-mpi --enable-timing CC=icc CXX=icpc
 *   - Please note that we used option --disable-shared, because HDF5
 *     library is only provided as a static library; if we do not use
 *     this option, linking with hdf5 fails (variable
 *     dependency_libs in the libtools file libhydroCpu.la is wrong) !!!
 *
 * - \c first \c run: you can try to execute a simple test (2d
 *     Orszag-Tang, 4 MPI processes for example); here is the bash script used to
 *     submit the job: run_job.sh
 *     \code
 *     #!/bin/bash
 *     #MSUB -r test_mh2d_cpu_mpi # Nom du job                
 *     #MSUB -p genXXXX               # mettre ici l'ID de votre projet
 *     #MSUB -n 4                     # Reservation de n process MPI
 *     #MSUB -N 2                     # reservation de N noeuds (2 devices GPU par noeud)
 *     #MSUB -T 1000                  # Limite de temps elapsed du job ici 1000s      
 *     #MSUB -o test_mhd2d_cpu_mpi.o  # Sortie standard
 *     #MSUB -e test_mhd2d_cpu_mpi.e  # Sortie d'erreur       
 *     #MSUB -@ pierre.kestener@cea.fr:end    # envoie un mail a l'adresse indiquee en fin de job
 *
 *     set -x
 *     mpirun ./euler_cpu_mpi --param ./orszag-tang2d_cpu_mpi.ini
 *     \endcode
 * Submit your job with:
 * msub ./run_job.sh
 *
 * - \c note: Don't forget to customize the parameter file: you need
 *     to change parameter outputDir to something related to your
 *     scratch directory (e.g. /scratch/cont003/pkestene/test_mhd2d_cpu_mpi/)
 *
 *
 * - \c note: library VTK is not available on Titane, so if you turn
 *     on VTK, it will generate VTK output files using our
 *     hand-written routines (not very optimal, but functionnal). You
 *     should prefer using HDF5 (which uses MPI collective IO properties).
 *
 * \section Contact Contact
 *
 * For any question, comment or advice, please feel free to email :
 *
 * \li Pierre Kestener, CEA / Maison de la Simulation : pierre.kestener at cea.fr
 * \li Sebastien Fromang, CEA / Service d'Astrophysique : sebastien.fromang at cea.fr
 *
 * \section Future Future developments
 *
 * <!-- \li add a MPI layer to target machines like TITANE at CCRT (done in
 * November 2010 for the Hydro scheme (split and unsplit)) -->
 *
 * <!-- \li include a MHD solver (done in August 2011) -->
 *
 * <!-- \li couple this code with D. Aubert and R. Teyssier CUDA-based
 * Particule-Mesh code (probably dropped for now) -->
 *
 * \li improve MHD solver: add dissipative terms (done in April 2012).
 *
 * \li improve GPU version : use CUDA streams to overlap memory transfert and 
 * computations.
 *
 *
 */
