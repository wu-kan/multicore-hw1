#!/bin/bash
#SBATCH -J WuK_scaffold
#SBATCH -p gpu_v100
#SBATCH -N 1
#SBATCH --exclusive

#module load cmake/3.14.3-gcc-4.8.5
#module load CUDA/10.1.2

mkdir -p sources/bin
cd sources/bin
rm -fr *

cmake ..  \
    -DCMAKE_C_FLAGS="-Ofast -fopenmp" \
    -DCMAKE_CXX_FLAGS="-Ofast -fopenmp" \
    -DCUDA_NVCC_FLAGS="-O3 -use_fast_math -Xcompiler -fopenmp"
make

cd ../..
sources/bin/main