All major changes to the GFCC library will be documented in this file.

## [v1.0](https://github.com/spec-org/gfcc/releases/tag/v1.0) (09-27-2019)

- Concurrent computation of frequency-dependent Green's function matrix elements and spectral function in the CCSD/GFCCSD level (enabled via MPI process groups)
- Support for multidimensional real/complex hybrid tensor contractions and slicing on both CPU and GPU
- On-the-fly Cholesky decomposition for atomic-orbital based two-electron integrals
- Direct inversion of the iterative subspace (DIIS) is customized and implemented as the default complex linear solver
- Gram-Schmidt orthogonalization for multidimensional complex tensors
- Model-order-reduction (MOR) procedure for complex linear systems
- Automatic resource throttling for various inexpensive operations
- Checkpointing (or restarting) calculation employing parallel IO operations for reading (writing) tensors from (to) disk

## [v1.1](https://github.com/spec-org/gfcc/releases/tag/v1.1) (06-04-2020)
- The default linear solver has been changed from DIIS to Generalized minimal residual method (GMRES)
- Load balancing across concurrent computation of different orbitals for a given frequency
- TAMM: Improved distributed tensor contraction performance on GPUs
