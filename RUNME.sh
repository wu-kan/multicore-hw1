#!/bin/bash
#SBATCH -J WuK_scaffold
#SBATCH -p gpu_v100
#SBATCH -N 1
#SBATCH --exclusive

spack load gcc@7.5.0
spack load cuda@10.1.243%gcc@7.5.0

rm -fr sources/bin
mkdir -p sources/bin
cd sources/bin

cmake .. \
    -DCMAKE_C_FLAGS=" -Ofast -fopenmp " \
    -DCMAKE_CXX_FLAGS=" -Ofast -fopenmp " \
    -DCUDA_NVCC_FLAGS=" -arch=sm_70 -O3 -use_fast_math -Xcompiler -Ofast -Xcompiler -fopenmp "
make

cd ../..
sources/bin/main
rm -fr sources/bin
