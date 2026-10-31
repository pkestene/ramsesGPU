**WARNING: This code is mostly deprecated. Have a look at [ppkMHD](https://github.com/pkestene/ppkMHD)**

[![DOI](https://zenodo.org/badge/42436229.svg)](https://zenodo.org/badge/latestdoi/42436229)

![mhd_mri 200x200](https://github.com/pkestene/ramsesGPU/blob/master/doc/mhd_mri_3d_gpu_Pm4_Re25000_double.gif)

[Magneto Rotational Instability](https://en.wikipedia.org/wiki/Magnetorotational_instability) simulation in a shearing box setup (800x1600x800) made in 2013 on [TGCC/CURIE](http://www-hpc.cea.fr/fr/complexe/tgcc-curie.htm) using 256 GPUs. Here [Reynolds number](https://en.wikipedia.org/wiki/Reynolds_number) is 25000 and [Prandtl number](https://en.wikipedia.org/wiki/Prandtl_number) is 4.

# RamsesGPU code

## RamsesGPU website

http://www.maisondelasimulation.fr/projects/RAMSES-GPU/html/index.html

- See doxygen-generated documentation in doc sub-directory

### Quickstart for building RAMSES-GPU using CMake

Default CUDA compilation flags can be passed to cmake using env variable CUDAFLAGS, or directly set CMAKE_CUDA_FLAGS on the configuration command line (see below).

```shell
# step 0:
git clone https://github.com/pkestene/ramsesGPU.git

# step 1:
cd ramsesGPU; mkdir build

# step 2:
cmake -DUSE_CUDA=ON -DUSE_MPI=ON -DCMAKE_CUDA_ARCHITECTURES=75 ..

# step 3:
make -j 8
```

Please, note that if you use a more recent version of cmake (>= 3.24), at step 2, you can use `-DCMAKE_CUDA_ARCHITECTURES=native` and cmake will auto-detect the GPU architecture on the build host.

You should get executable *ramsesGPU_mpi_cuda*. Explore other flag using the ccmake user interface.


This will build the monoCPU / monoGPU version of the programme to solve hydro/MHD problems. Executable are located in src subdirectory and named `ramsesGPU_cpu` / `ramsesGPU_cuda`

Execute a test run:

```bash
cd src
./ramsesGPU_cuda --param ../data/jet2d_gpu.ini

```

This will start a Hydrodynamics 2D jet simulation run. Output files can be
in multiple file format (VTK, Hdf5, ...).

To visualize Hdf5 output, just run:

``` bash
paraview --data=./jet2d_gpu.xmf
```

Contact, questions, comments:

pierre.kestener at cea.fr
