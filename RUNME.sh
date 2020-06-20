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
    -DCMAKE_C_FLAGS="-O3 -fopenmp" \
    -DCMAKE_CXX_FLAGS="-O3 -fopenmp" \
    -DCUDA_NVCC_FLAGS="-O3 -Xcompiler -fopenmp"
make

cd ../..
sources/bin/main