#PBS -N scaffold
#PBS -l nodes=1:ppn=32:gpus=1
#PBS -j oe
#PBS -q gpu

source /public/software/profile.d/cuda10.0.sh
cd $PBS_O_WORKDIR

mkdir -p sources/bin
cd sources/bin
rm -fr *

cmake ..  \
    -DCMAKE_C_FLAGS="-Ofast -fopenmp" \
    -DCMAKE_CXX_FLAGS="-Ofast -fopenmp" \
    -DCUDA_NVCC_FLAGS="-O3 -use_fast_math -ftz=true -Xcompiler -fopenmp"
make

cd ../..
sources/bin/main