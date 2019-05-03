[![DOI](https://zenodo.org/badge/42436229.svg)](https://zenodo.org/badge/latestdoi/42436229)

![mhd_mri 200x200](https://github.com/pkestene/ramsesGPU/blob/master/doc/mhd_mri_3d_gpu_Pm4_Re25000_double.gif)

[Magneto Rotational Instability](https://en.wikipedia.org/wiki/Magnetorotational_instability) simulation in a shearing box setup (800x1600x800) made in 2013 on [TGCC/CURIE](http://www-hpc.cea.fr/fr/complexe/tgcc-curie.htm) using 256 GPUs. Here [Reynolds number](https://en.wikipedia.org/wiki/Reynolds_number) is 25000 and [Prandtl number](https://en.wikipedia.org/wiki/Prandtl_number) is 4.

# RamsesGPU code

## RamsesGPU website

http://www.maisondelasimulation.fr/projects/RAMSES-GPU/html/index.html

- See doxygen-generated documentation in doc sub-directory

- Quickstart for building RAMSES-GPU using CMake (recommended)

0. git clone https://github.com/pkestene/ramsesGPU.git
1. cd ramsesGPU; mkdir build
2. cmake -DUSE_GPU=ON -DUSE_MPI=ON ..
3. make

You should get executable *ramsesGPU_mpi_cuda*. Explore other flag using the ccmake user interface.

- Quickstart for building RAMSES-GPU using autotools (deprecated)

0. make sure to have up-to-date autotools on you build system (autoconf, automake, libtool, m4); then run `sh autogen.sh`
1. configure --with-cuda=<path to CUDA toolkit root directory> 
2. make (or make -j N to speed-up compilation time; you might need to execute make several times when using option -j)

Note: make sure to have CUDA toolkit installed, and environment variables PATH and LD_LIBRARY_PATH correctly set.

This will build the monoCPU / monoGPU version of the programme to solve hydro/MHD problems. Executable are located in src subdirectory and named euler_cpu / euler_gpu

Execute a test run:
	
```bash
	cd src
	./euler_gpu --param ../data/jet2d_gpu.ini

```

This will start a Hydrodynamics 2D jet simulation run. Output files can be
in multiple file format (VTK, Hdf5, ...).

To visualize Hdf5 output, just run:

``` bash

	paraview --data=./jet2d_gpu.xmf
```


Contact, questions, comments:

pierre.kestener at cea.fr
