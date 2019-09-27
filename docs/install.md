
The prerequisites needed to build:

- autotools
- cmake >= 3.13
- MPI Library
- C++17 compiler
- CUDA >= 10.1 (only if building with GPU support)

`MACOSX`: We recommend using brew to install the prerequisites:  
- `brew install gcc openmpi cmake wget autoconf automake`

Supported Compilers
--------------------
- GCC versions >= 8.x
- LLVM Clang >= 7.x (Tested on Linux Only)
- `MACOSX`: We only support brew installed `GCC`, AppleClang is not supported.


Supported Configurations
-------------------------
- The following configurations are recommended since they are tested and are known to work:
  - GCC versions >= 8.x + OpenMPI-2.x/MPICH-3.x built using corresponding gcc versions.
  - LLVM Clang versions >= 7.x + OpenMPI-2.x/MPICH-3.x 


```
export GFCC_INSTALL_PATH=$HOME/gfcc_install

git clone https://github.com/spec-org/gfcc.git
cd contrib/CMakeBuild
mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=$GFCC_INSTALL_PATH \
-DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ -DCMAKE_Fortran_COMPILER=gfortran ..

make install
```

General TAMM build using GCC
----------------------------
```
cd contrib/TAMM
mkdir build && cd build

cmake \
-DCMAKE_INSTALL_PREFIX=$GFCC_INSTALL_PATH \
-DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ -DCMAKE_Fortran_COMPILER=gfortran ..

#CUDA Options
[-DNWX_CUDA=ON] #OFF by Default

make -j3
make install
```

Building the GFCC library
-------------------------
```
- At the top level of this repository

mkdir build && cd build

cmake \
-DCMAKE_INSTALL_PREFIX=$GFCC_INSTALL_PATH \
-DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ -DCMAKE_Fortran_COMPILER=gfortran ..

make -j2
```

------------------
## RUNNING THE CODE
------------------
From the build folder, run:  
`mpirun -n 4 ./test_stage/$GFCC_INSTALL_PATH/gfcc_install/tests/GF_CCSD_CS ../tests/co.nwx`


--------------------------------------------------------
# Advanced Build Options for TAMM/libGFCC
--------------------------------------------------------

Build using GCC+MKL
----------------------------

Set `GFCC_INSTALL_PATH` and `INTEL_ROOT` accordingly

```
export GFCC_INSTALL_PATH=$HOME/gfcc_install
export INTEL_ROOT=/opt/intel/compilers_and_libraries_2019.0.117

export MKL_INC=$INTEL_ROOT/linux/mkl/include
export MKL_LIBS=$INTEL_ROOT/linux/mkl/lib/intel64

export TAMM_BLASLIBS="$MKL_LIBS/libmkl_intel_ilp64.a;$MKL_LIBS/libmkl_lapack95_ilp64.a;$MKL_LIBS/libmkl_blas95_ilp64.a;$MKL_LIBS/libmkl_intel_thread.a;$MKL_LIBS/libmkl_core.a;$INTEL_ROOT/linux/compiler/lib/intel64/libiomp5.a;-lpthread;-ldl"

cmake \
-DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ -DCMAKE_Fortran_COMPILER=gfortran \
-DCBLAS_INCLUDE_DIRS=$MKL_INC \
-DLAPACKE_INCLUDE_DIRS=$MKL_INC \
-DCMAKE_INSTALL_PREFIX=$GFCC_INSTALL_PATH \
-DCBLAS_LIBRARIES=$TAMM_BLASLIBS \
-DLAPACKE_LIBRARIES=$TAMM_BLASLIBS ..
```

```
make -j3
make install
```

Build instructions for Summit (using GCC+ESSL)
----------------------------------------------

```
module load gcc/8.1.1
module load cmake/3.14.2
module load spectrum-mpi/10.3.0.1-20190611
module load essl/6.1.0-2
module load cuda/10.1.105
```

```
The following paths may need to be adjusted if the modules change:

export GFCC_INSTALL_PATH=$HOME/gfcc_install
export ESSL_INC=/sw/summit/essl/6.1.0-2/essl/6.1/include
export TAMM_BLASLIBS="/sw/summit/essl/6.1.0-2/essl/6.1/lib64/libesslsmp6464.so"
```
```
 cmake \
-DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ -DCMAKE_Fortran_COMPILER=gfortran \
-DCBLAS_INCLUDE_DIRS=$ESSL_INC \
-DLAPACKE_INCLUDE_DIRS=$ESSL_INC \
-DCMAKE_INSTALL_PREFIX=$GFCC_INSTALL_PATH \
-DCBLAS_LIBRARIES=$TAMM_BLASLIBS \
-DLAPACKE_LIBRARIES=$TAMM_BLASLIBS \
-DTAMM_CXX_FLAGS="-mcpu=power9" \
-DBLIS_CONFIG=power9 ..

To enable CUDA build, add -DNWX_CUDA=ON

```


Build instructions for Cori
----------------------------

```
module unload PrgEnv-intel/6.0.5
module load PrgEnv-gnu/6.0.5
module swap gcc/8.2.0 
module swap craype/2.5.18
module swap cray-mpich/7.7.6 
module load cmake/3.14.4 
module load cuda/10.1.168

```

- `NOTE:` CMakeBuild repository should be built with the following compiler options.
  - Remove the compiler options from the cmake line or change them to:  
 -DCMAKE_C_COMPILER=cc -DCMAKE_CXX_COMPILER=CC -DCMAKE_Fortran_COMPILER=ftn

 
```
export CRAYPE_LINK_TYPE=dynamic

export GFCC_INSTALL_PATH=$HOME/gfcc_install
export INTEL_ROOT=/opt/intel/compilers_and_libraries_2019.3.199

export MKL_INC=$INTEL_ROOT/linux/mkl/include
export MKL_LIBS=$INTEL_ROOT/linux/mkl/lib/intel64
export TAMM_BLASLIBS="$MKL_LIBS/libmkl_intel_ilp64.a;$MKL_LIBS/libmkl_lapack95_ilp64.a;$MKL_LIBS/libmkl_blas95_ilp64.a;$MKL_LIBS/libmkl_gnu_thread.a;$MKL_LIBS/libmkl_core.a;-lgomp;-lpthread;-ldl"

cmake -DCBLAS_INCLUDE_DIRS=$MKL_INC \
-DLAPACKE_INCLUDE_DIRS=$MKL_INC \
-DCMAKE_INSTALL_PREFIX=$GFCC_INSTALL_PATH \
-DCBLAS_LIBRARIES=$TAMM_BLASLIBS \
-DLAPACKE_LIBRARIES=$TAMM_BLASLIBS ..

To enable CUDA build, add -DNWX_CUDA=ON

```
Build instructions for Mac OS
----------------------------

```
brew install gcc openmpi cmake wget autoconf automake

export FC=gfortran-9
export CC=gcc-9
export CXX=g++-9
export GFCC_INSTALL_PATH=$HOME/gfcc_install

git clone https://github.com/spec-org/gfcc.git

cd gfcc

cd contrib/CMakeBuild
mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=$GFCC_INSTALL_PATH ..
make install

cd ../../TAMM
mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=$GFCC_INSTALL_PATH ..
make -j3 install

cd ../../..
mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=$GFCC_INSTALL_PATH ..
make -j2 install
```

Build instructions for Ubuntu Bionic 18.04
----------------------------

```
export FC=gfortran-8
export CC=gcc-8
export CXX=g++-8
export GFCC_INSTALL_PATH=$HOME/gfcc_install

sudo apt install g++-8 gcc-8 gfortran-8 openmpi-dev

git clone https://github.com/spec-org/gfcc.git

cd gfcc

curl -LJO https://github.com/Kitware/CMake/releases/download/v3.15.3/cmake-3.15.3-Linux-x86_64.tar.gz
tar xzf cmake-3.15.3-Linux-x86_64.tar.gz
export PATH=`pwd`/cmake-3.15.3-Linux-x86_64/bin:$PATH

cd contrib/CMakeBuild
mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=$GFCC_INSTALL_PATH ..
make install

cd ../../TAMM
mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=$GFCC_INSTALL_PATH ..
make -j3 install

cd ../../..
mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=$GFCC_INSTALL_PATH ..
make -j2 install
```
