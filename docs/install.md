
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


# Build Instructions

```
export GFCC_SRC=$HOME/gfcc_src
export GFCC_INSTALL_PATH=$HOME/gfcc_install
git clone https://github.com/spec-org/gfcc.git $GFCC_SRC
```

Step 1: Setup CMakeBuild
-------------------------
```
cd $GFCC_SRC/contrib/CMakeBuild
mkdir build && cd build

CC=gcc CXX=g++ FC=gfortran cmake -DCMAKE_INSTALL_PREFIX=$GFCC_INSTALL_PATH ..

make install
```

Step 2: General TAMM build using GCC
------------------------------------
```
cd $GFCC_SRC/contrib/TAMM
mkdir build && cd build

CC=gcc CXX=g++ FC=gfortran cmake -DCMAKE_INSTALL_PREFIX=$GFCC_INSTALL_PATH ..

#CUDA Options
[-DNWX_CUDA=ON] #OFF by Default

# make step takes a while, please use as many cores as possible
make -j3
make install
```

Step 3: Building the GFCC library
---------------------------------
```
cd $GFCC_SRC
mkdir build && cd build

CC=gcc CXX=g++ FC=gfortran cmake -DCMAKE_INSTALL_PREFIX=$GFCC_INSTALL_PATH ..

make -j2
```

Step 4: Running the GFCC code
------------------------------
`cd $GFCC_SRC/build`   
`mpirun -n 4 ./test_stage/$GFCC_INSTALL_PATH/tests/GF_CCSD_CS ../tests/co.nwx`


-------------------------------------------------------------------
# Advanced Build Options for TAMM/libGFCC (applies to Steps 2 & 3)
-------------------------------------------------------------------

Build using GCC+MKL
----------------------------

Set `GFCC_INSTALL_PATH` and `INTEL_ROOT` accordingly

```
export GFCC_INSTALL_PATH=$HOME/gfcc_install
export INTEL_ROOT=/opt/intel/compilers_and_libraries_2019.0.117

export MKL_INC=$INTEL_ROOT/linux/mkl/include
export MKL_LIBS=$INTEL_ROOT/linux/mkl/lib/intel64

export TAMM_BLASLIBS="$MKL_LIBS/libmkl_intel_ilp64.a;$MKL_LIBS/libmkl_lapack95_ilp64.a;$MKL_LIBS/libmkl_blas95_ilp64.a;$MKL_LIBS/libmkl_intel_thread.a;$MKL_LIBS/libmkl_core.a;$INTEL_ROOT/linux/compiler/lib/intel64/libiomp5.a;-lpthread;-ldl"

CC=gcc CXX=g++ FC=gfortran cmake \
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
CC=gcc CXX=g++ FC=gfortran cmake \
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

- `NOTE:` CMakeBuild (in Step 1) should also be built with the following compiler options.
  - Remove the compiler options from the cmake line or change them to:  
    CC=cc CXX=CC FC=ftn 

 
```
export CRAYPE_LINK_TYPE=dynamic

export GFCC_INSTALL_PATH=$HOME/gfcc_install
export INTEL_ROOT=/opt/intel/compilers_and_libraries_2019.3.199

export MKL_INC=$INTEL_ROOT/linux/mkl/include
export MKL_LIBS=$INTEL_ROOT/linux/mkl/lib/intel64
export TAMM_BLASLIBS="$MKL_LIBS/libmkl_intel_ilp64.a;$MKL_LIBS/libmkl_lapack95_ilp64.a;$MKL_LIBS/libmkl_blas95_ilp64.a;$MKL_LIBS/libmkl_gnu_thread.a;$MKL_LIBS/libmkl_core.a;-lgomp;-lpthread;-ldl"


CC=cc CXX=CC FC=ftn cmake -DCBLAS_INCLUDE_DIRS=$MKL_INC \
-DLAPACKE_INCLUDE_DIRS=$MKL_INC \
-DCMAKE_INSTALL_PREFIX=$GFCC_INSTALL_PATH \
-DCBLAS_LIBRARIES=$TAMM_BLASLIBS \
-DLAPACKE_LIBRARIES=$TAMM_BLASLIBS ..

To enable CUDA build, add -DNWX_CUDA=ON

```
Build instructions for Mac OS
-------------------------------

```
brew install gcc openmpi cmake wget autoconf automake

export GFCC_SRC=$HOME/gfcc_src
export GFCC_INSTALL_PATH=$HOME/gfcc_install
git clone https://github.com/spec-org/gfcc.git GFCC_SRC

cd $GFCC_SRC/contrib/CMakeBuild
mkdir build && cd build
CC=gcc-8 CXX=g++-8 FC=gfortran-8 cmake -DCMAKE_INSTALL_PREFIX=$GFCC_INSTALL_PATH ..
make install

cd $GFCC_SRC/contrib/TAMM
mkdir build && cd build
CC=gcc-8 CXX=g++-8 FC=gfortran-8 cmake -DCMAKE_INSTALL_PREFIX=$GFCC_INSTALL_PATH ..
make -j3 
make install

cd $GFCC_SRC
mkdir build && cd build
CC=gcc-8 CXX=g++-8 FC=gfortran-8 cmake -DCMAKE_INSTALL_PREFIX=$GFCC_INSTALL_PATH ..
make -j2 install
```

Build instructions for Ubuntu Bionic 18.04
------------------------------------------

```
sudo apt install g++-8 gcc-8 gfortran-8 openmpi-dev

curl -LJO https://github.com/Kitware/CMake/releases/download/v3.15.3/cmake-3.15.3-Linux-x86_64.tar.gz
tar xzf cmake-3.15.3-Linux-x86_64.tar.gz
export PATH=`pwd`/cmake-3.15.3-Linux-x86_64/bin:$PATH

export GFCC_SRC=$HOME/gfcc_src
export GFCC_INSTALL_PATH=$HOME/gfcc_install
git clone https://github.com/spec-org/gfcc.git GFCC_SRC

cd $GFCC_SRC/contrib/CMakeBuild
mkdir build && cd build
CC=gcc-8 CXX=g++-8 FC=gfortran-8 cmake -DCMAKE_INSTALL_PREFIX=$GFCC_INSTALL_PATH ..
make install

cd $GFCC_SRC/contrib/TAMM
mkdir build && cd build
CC=gcc-8 CXX=g++-8 FC=gfortran-8 cmake -DCMAKE_INSTALL_PREFIX=$GFCC_INSTALL_PATH ..
make -j3 install

cd $GFCC_SRC
mkdir build && cd build
CC=gcc-8 CXX=g++-8 FC=gfortran-8 cmake -DCMAKE_INSTALL_PREFIX=$GFCC_INSTALL_PATH ..
make -j2 install
```
